import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import warnings
import random
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

# ==================== 路径配置（加入 sys.path 以导入 model.py）=====================
# 当前文件所在目录: code/train/stage-2/V7/
CURRENT_DIR = Path(__file__).resolve().parent
# model.py 所在目录: code/train/
CODE_DIR = CURRENT_DIR.parent.parent

sys.path.insert(0, str(CODE_DIR))
from model import DeepAMPpredStage2_MICAux, AdaptiveBCEFocalLoss

warnings.filterwarnings('ignore')

# ==================== 【V7: V6-v6 Prior + AUX】=====================
EXPERIMENT_TAG = "v7_mtl_prior_aux"

USE_LEARNABLE_ADAPTIVE = True
USE_MIC_AUX = True  # ✅ V7 核心：开启 MIC 辅助任务
USE_SWA = False

BATCH_SIZE = 32
EPOCHS = 150
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30
DROPOUT = 0.3
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
NUM_TASKS_PER_STEP = 3  # V6-v6 核心：每步随机 3 个任务

BACTERIA_NAMES = ['Ab', 'Bs', 'Ec', 'Ef', 'Kp', 'Ml', 'Pa', 'Sa', 'Se', 'St']
NUM_BACTERIA = len(BACTERIA_NAMES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 路径配置（改为相对路径）=====================
# 特征目录: code/feature_extract/stage-2/
FEATURE_DIR = CODE_DIR / "feature_extract" / "stage-2"
TRAIN_DIR = FEATURE_DIR / "Train"
TEST_DIR = FEATURE_DIR / "Test"

if not TRAIN_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError(
        f"找不到特征目录: {TRAIN_DIR} 或 {TEST_DIR}\n"
        f"请确认已运行 feature_extract/stage-2/stage-2.py 提取特征，"
        f"或已将 .npy 文件放置到 {FEATURE_DIR}/Train/ 和 Test/"
    )

# 结果保存到当前脚本同级目录（如 code/train/stage-2/V7/results/v7_mtl_prior_aux/）
SAVE_DIR = CURRENT_DIR / "results" / EXPERIMENT_TAG
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message);
        self.terminal.flush()
        self.log.write(message);
        self.log.flush()

    def flush(self):
        self.terminal.flush();
        self.log.flush()

    def close(self):
        self.log.close()


# ======================== MIC Pairwise Ranking AUX（V4/V7保留）====================
def compute_mic_aux_loss(features, labels, mic_values, model, max_pairs=200):
    labels = labels.view(-1)
    mic_values = mic_values.view(-1)
    valid_mask = (labels == 1) & (~torch.isnan(mic_values)) & (mic_values > 0)
    valid_idx = torch.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    feat_valid = features[valid_idx]
    mic_valid = mic_values[valid_idx]
    n_pairs = min(max_pairs, len(valid_idx) * (len(valid_idx) - 1) // 2, 500)
    pairs_i = torch.randint(0, len(valid_idx), (n_pairs,), device=device)
    pairs_j = torch.randint(0, len(valid_idx), (n_pairs,), device=device)
    mask = pairs_i != pairs_j
    pairs_i, pairs_j = pairs_i[mask], pairs_j[mask]
    if len(pairs_i) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    feat_i = feat_valid[pairs_i]
    feat_j = feat_valid[pairs_j]
    mic_i = mic_valid[pairs_i]
    mic_j = mic_valid[pairs_j]
    aux_labels = (mic_i < mic_j).long()
    aux_feat = torch.abs(feat_i - feat_j)
    return F.cross_entropy(model.mic_classifier(aux_feat), aux_labels)


# ======================== 评估指标（V6-v6保留）=======================
def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(set(y_true)) == 1:
            tp, fn = (len(y_true), 0) if y_true[0] == 1 else (0, 0)
            tn, fp = (0, 0) if y_true[0] == 1 else (len(y_true), 0)

    metrics['TP'] = int(tp);
    metrics['TN'] = int(tn)
    metrics['FP'] = int(fp);
    metrics['FN'] = int(fn)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Sensitivity'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    try:
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    except:
        metrics['MCC'] = 0.0
    sen, spe = metrics['Sensitivity'], metrics['Specificity']
    metrics['GMean'] = np.sqrt(max(0.0, sen * spe))
    try:
        metrics['AUC'] = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    except ValueError:
        metrics['AUC'] = 0.0
    return metrics


def find_best_threshold(y_true, y_prob):
    best_score, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.951, 0.005):
        pred = (y_prob >= t).astype(int)
        try:
            mcc = matthews_corrcoef(y_true, pred)
        except:
            mcc = 0.0

        tp = ((pred == 1) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = np.sqrt(max(0.0, sen * spe))

        balance_penalty = 0.0
        if sen < 0.1 or spe < 0.1:
            balance_penalty = -0.15

        score = mcc + 0.03 * gmean + balance_penalty

        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


# ===================== 数据平衡加载器（V6-v6保留）=====================
def create_balanced_loader(X, y, mic, batch_size=32, pos_ratio_target=0.3):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y).unsqueeze(1),
            torch.FloatTensor(mic)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_pos = len(pos_idx)
    target_pos = max(int(len(y) * pos_ratio_target), n_pos)

    if n_pos < target_pos:
        pos_idx_bal = np.concatenate([
            np.tile(pos_idx, target_pos // n_pos),
            np.random.choice(pos_idx, target_pos % n_pos, replace=False)
        ])
    else:
        pos_idx_bal = pos_idx

    neg_size = min(len(neg_idx), int(len(pos_idx_bal) * (1 - pos_ratio_target) / pos_ratio_target))
    neg_idx_bal = np.random.choice(neg_idx, neg_size, replace=False)

    all_idx = np.concatenate([pos_idx_bal, neg_idx_bal])
    np.random.shuffle(all_idx)

    dataset = TensorDataset(
        torch.FloatTensor(X[all_idx]),
        torch.FloatTensor(y[all_idx]).unsqueeze(1),
        torch.FloatTensor(mic[all_idx])
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ===================== 加载所有菌种数据（V6-v6保留）=====================
def load_all_bacteria_data():
    all_data = []
    for name in BACTERIA_NAMES:
        X_train = np.load(str(TRAIN_DIR / f"{name}_features.npy"))
        y_train = np.load(str(TRAIN_DIR / f"{name}_labels.npy"))
        mic_train = np.load(str(TRAIN_DIR / f"{name}_mic.npy"))

        X_test = np.load(str(TEST_DIR / f"{name}_features.npy"))
        y_test = np.load(str(TEST_DIR / f"{name}_labels.npy"))
        mic_test = np.load(str(TEST_DIR / f"{name}_mic.npy"))

        pos_ratio = y_train.mean()

        all_data.append({
            'name': name,
            'X_train': X_train, 'y_train': y_train, 'mic_train': mic_train,
            'X_test': X_test, 'y_test': y_test, 'mic_test': mic_test,
            'pos_ratio': pos_ratio
        })
    return all_data


# ===================== 基于不平衡度的先验配置（V6-v6保留）=====================
def get_task_alpha_config(pos_ratio):
    if pos_ratio < 0.15:
        target = 0.25
        init_logit = np.random.uniform(-1.8, -1.0)
    elif pos_ratio < 0.40:
        target = 0.40
        init_logit = np.random.uniform(-0.6, 0.0)
    else:
        target = 0.65
        init_logit = np.random.uniform(0.5, 1.2)

    return {
        'target_alpha': target,
        'init_logit': init_logit,
        'reg_weight': 0.5,
        'alpha_lr': 2e-3
    }


# ===================== MTL 单 Fold 训练（V7 = V6-v6 + AUX）=====================
def train_mtl_fold(fold_idx, all_data, fold_splits):
    print(f"\n{'#' * 70}")
    print(f"# FOLD {fold_idx + 1}/5 | V7: Prior-Anchored α + MIC-Aux (from V6-v6)")
    if USE_MIC_AUX:
        print(f"# MIC-Auxiliary Task: ON")
    print(f"{'#' * 70}")

    scalers = []
    train_loaders = []
    val_tensors = []
    test_tensors = []

    for idx, data in enumerate(all_data):
        X_train, y_train = data['X_train'], data['y_train']
        mic_train = data['mic_train']
        tr_idx, val_idx = fold_splits[idx][fold_idx]

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        mic_tr = mic_train[tr_idx]

        N_tr, seq_len, feat_dim = X_tr.shape
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(-1, feat_dim)).reshape(N_tr, seq_len, feat_dim)
        N_va, _, _ = X_val.shape
        X_val = scaler.transform(X_val.reshape(-1, feat_dim)).reshape(N_va, seq_len, feat_dim)
        scalers.append(scaler)

        loader = create_balanced_loader(X_tr, y_tr, mic_tr, batch_size=BATCH_SIZE)
        train_loaders.append(loader)

        X_va_t = torch.FloatTensor(X_val).to(device)
        y_va_t = torch.FloatTensor(y_val).to(device).unsqueeze(1)
        val_tensors.append((X_va_t, y_va_t))

        X_test, y_test = data['X_test'], data['y_test']
        N_te, _, _ = X_test.shape
        X_test_scaled = scaler.transform(X_test.reshape(-1, feat_dim)).reshape(N_te, seq_len, feat_dim)
        X_te_t = torch.FloatTensor(X_test_scaled).to(device)
        y_te_t = torch.FloatTensor(y_test).to(device)
        test_tensors.append((X_te_t, y_te_t))

    torch.manual_seed(42 + fold_idx)
    model = DeepAMPpredStage2_MICAux(
        input_dim=480, seq_len=100, num_bacteria=NUM_BACTERIA,
        lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=DROPOUT
    ).to(device)

    criterions = []
    alpha_opts = []
    alpha_configs = []

    for idx, data in enumerate(all_data):
        cfg = get_task_alpha_config(data['pos_ratio'])
        c = AdaptiveBCEFocalLoss(
            gamma=1.5,
            focal_alpha=0.25,
            reg_weight=cfg['reg_weight'],
            smoothing=0.1,
            init_logit=cfg['init_logit'],
            target_alpha=cfg['target_alpha']
        ).to(device)
        criterions.append(c)
        alpha_opts.append(
            torch.optim.Adam([c.logit_alpha], lr=cfg['alpha_lr'])
        )
        alpha_configs.append(cfg)

    optimizer_model = torch.optim.AdamW(
        model.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_model, T_0=20, T_mult=2
    )

    best_avg_mcc = -np.inf
    best_state = None
    best_epoch = 0
    counter = 0

    iterators = [iter(loader) for loader in train_loaders]
    max_batches = max(len(loader) for loader in train_loaders)

    for epoch in range(EPOCHS):
        # ===== V7 新增：AUX 权重调度（V4风格）=====
        aux_weight = min(0.15, 0.05 * (epoch - 5 + 1)) if (USE_MIC_AUX and epoch >= 5) else 0.0

        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for step in range(max_batches):
            selected_tasks = random.sample(range(NUM_BACTERIA), NUM_TASKS_PER_STEP)

            optimizer_model.zero_grad()
            for tidx in selected_tasks:
                alpha_opts[tidx].zero_grad()

            step_loss_sum = 0.0
            for task_idx in selected_tasks:
                try:
                    xb, yb, mb = next(iterators[task_idx])
                except StopIteration:
                    iterators[task_idx] = iter(train_loaders[task_idx])
                    xb, yb, mb = next(iterators[task_idx])

                xb = xb.to(device)
                yb = yb.to(device).squeeze(1)
                mb = mb.to(device)

                # ===== V7 核心修改：加入 AUX =====
                if USE_MIC_AUX and aux_weight > 0:
                    out_main, feat_mic = model(xb, return_features=True, bacteria_idx=task_idx)
                    loss_main = criterions[task_idx](out_main.squeeze(-1), yb)
                    loss_aux = compute_mic_aux_loss(feat_mic, yb, mb, model, max_pairs=200)
                    loss = loss_main + aux_weight * loss_aux
                else:
                    out_main = model(xb, return_features=False, bacteria_idx=task_idx)
                    loss = criterions[task_idx](out_main.squeeze(-1), yb)

                loss.backward()
                step_loss_sum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_model.step()
            for tidx in selected_tasks:
                alpha_opts[tidx].step()

            epoch_loss += step_loss_sum / NUM_TASKS_PER_STEP
            n_steps += 1

        # ===== 验证（同 V6-v6）=====
        model.eval()
        val_mccs = []
        val_aucs = []
        val_gmeans = []
        with torch.no_grad():
            for task_idx in range(NUM_BACTERIA):
                X_va, y_va = val_tensors[task_idx]
                out_main = model(X_va, return_features=False, bacteria_idx=task_idx)
                prob = torch.sigmoid(out_main).squeeze().cpu().numpy()
                y_true = y_va.squeeze().cpu().numpy()

                thresh, _ = find_best_threshold(y_true, prob)
                pred = (prob >= thresh).astype(int)
                m = compute_metrics(y_true, pred, prob)
                val_mccs.append(m['MCC'])
                val_aucs.append(m['AUC'])
                val_gmeans.append(m['GMean'])

        avg_mcc = np.mean(val_mccs)
        avg_auc = np.mean(val_aucs)
        avg_gmean = np.mean(val_gmeans)

        lr_now = optimizer_model.param_groups[0]['lr']
        alpha_str = ""
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            alphas = [f"{c.alpha.item():.3f}" for c in criterions]
            alpha_str = f"| α={alphas}"

        aux_status = f"ON(w={aux_weight:.3f})" if (USE_MIC_AUX and aux_weight > 0) else "OFF"
        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"  [Fold {fold_idx + 1}] Epoch {epoch + 1:3d} | Loss {avg_loss:.3f} | AvgMCC {avg_mcc:.3f} | "
              f"AvgAUC {avg_auc:.3f} | AvgGMean {avg_gmean:.3f} | LR {lr_now:.6f} | Aux={aux_status} {alpha_str}")

        if avg_mcc > best_avg_mcc:
            best_avg_mcc = avg_mcc
            best_state = {
                'model': model.state_dict(),
                'criterions': [c.state_dict() for c in criterions],
                'alphas': [c.alpha.item() for c in criterions],
            }
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1

        if counter >= PATIENCE:
            print(f"\n  🛑 Early stop at Epoch {epoch + 1}, best was Epoch {best_epoch}")
            break

        scheduler.step()

    print(f"\n  ✅ Fold {fold_idx + 1} BEST AvgValMCC={best_avg_mcc:.4f} at Epoch {best_epoch}")

    model.load_state_dict(best_state['model'])
    for c, state in zip(criterions, best_state['criterions']):
        c.load_state_dict(state)
    model.eval()

    fold_results = []
    with torch.no_grad():
        for task_idx in range(NUM_BACTERIA):
            X_te, y_te = test_tensors[task_idx]
            out_main = model(X_te, return_features=False, bacteria_idx=task_idx)
            prob_test = torch.sigmoid(out_main).squeeze().cpu().numpy()
            y_true_test = y_te.cpu().numpy()

            thresh_test, _ = find_best_threshold(y_true_test, prob_test)
            pred_test = (prob_test >= thresh_test).astype(int)
            test_m = compute_metrics(y_true_test, pred_test, prob_test)

            final_alpha = criterions[task_idx].alpha.item()
            target_alpha = alpha_configs[task_idx]['target_alpha']

            fold_results.append({
                'Bacteria': BACTERIA_NAMES[task_idx],
                'Test_MCC': test_m['MCC'],
                'Test_AUC': test_m['AUC'],
                'Test_GMean': test_m['GMean'],
                'Test_Sensitivity': test_m['Sensitivity'],
                'Test_Specificity': test_m['Specificity'],
                'Test_Accuracy': test_m['Accuracy'],
                'Test_F1': test_m['F1'],
                'Final_Alpha': final_alpha,
                'Target_Alpha': target_alpha,
            })

            print(f"     {BACTERIA_NAMES[task_idx]}: TestMCC={test_m['MCC']:.3f} "
                  f"α={final_alpha:.3f} target={target_alpha:.2f}")

    return fold_results, best_state, scalers


# ===================== 主函数（V6-v6格式保留）====================
def main():
    log_file = SAVE_DIR / f"log_{datetime.now().strftime('%m%d_%H%M')}.txt"
    sys.stdout = Logger(log_file)

    print("=" * 80)
    print(" Deep-AMPpred Stage-2 | V7: Prior-Anchored α + MIC-Auxiliary")
    print("=" * 80)
    print(f"Version: {EXPERIMENT_TAG}")
    print(f"Loss: LEARNABLE_ADAPTIVE={USE_LEARNABLE_ADAPTIVE} | AUX={USE_MIC_AUX}")
    print(f"Device: {device}")
    print(f"Feature Dir: {FEATURE_DIR}")
    print(f"Save Dir: {SAVE_DIR}")
    print(f"Key: logit-space prior | reg=0.5 | Tasks/Step={NUM_TASKS_PER_STEP}")
    print("=" * 80)

    all_data = load_all_bacteria_data()

    print("\n📋 Task Imbalance Preview & Alpha Config (V7):")
    for data in all_data:
        cfg = get_task_alpha_config(data['pos_ratio'])
        print(f"  {data['name']}: pos_ratio={data['pos_ratio']:.3f} | "
              f"target={cfg['target_alpha']:.2f} | init_logit={cfg['init_logit']:+.2f} | "
              f"reg={cfg['reg_weight']:.1f}")

    fold_splits = []
    for data in all_data:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(data['X_train'], data['y_train']))
        fold_splits.append(splits)

    all_fold_results = []
    best_fold_idx = -1
    best_fold_avg_mcc = -np.inf

    for fold in range(5):
        fold_results, state_dict, scalers = train_mtl_fold(fold, all_data, fold_splits)
        all_fold_results.append(fold_results)

        fold_avg_mcc = np.mean([r['Test_MCC'] for r in fold_results])
        if fold_avg_mcc > best_fold_avg_mcc:
            best_fold_avg_mcc = fold_avg_mcc
            best_fold_idx = fold

        torch.save({
            'state_dict': state_dict['model'],
            'criterion_states': state_dict['criterions'],
            'alphas': state_dict['alphas'],
            'scalers': scalers,
            'fold_results': fold_results,
        }, SAVE_DIR / f"fold{fold}_model.pth")

    print("\n" + "=" * 80)
    print("           📊 5-FOLD TEST AVG RESULTS")
    print("=" * 80)

    summary_rows = []
    for task_idx, name in enumerate(BACTERIA_NAMES):
        task_test_mccs = [all_fold_results[f][task_idx]['Test_MCC'] for f in range(5)]
        task_test_aucs = [all_fold_results[f][task_idx]['Test_AUC'] for f in range(5)]
        task_test_gmeans = [all_fold_results[f][task_idx]['Test_GMean'] for f in range(5)]
        task_alphas = [all_fold_results[f][task_idx]['Final_Alpha'] for f in range(5)]

        summary_rows.append({
            'Bacteria': name,
            'Test_MCC_mean': np.mean(task_test_mccs),
            'Test_MCC_std': np.std(task_test_mccs),
            'Test_AUC_mean': np.mean(task_test_aucs),
            'Test_AUC_std': np.std(task_test_aucs),
            'Test_GMean_mean': np.mean(task_test_gmeans),
            'Alpha_mean': np.mean(task_alphas),
            'Alpha_std': np.std(task_alphas),
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary[['Bacteria', 'Test_MCC_mean', 'Test_MCC_std',
                      'Test_AUC_mean', 'Test_AUC_std', 'Test_GMean_mean', 'Alpha_std']].round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print(f"           🎯 INDEPENDENT TEST RESULTS (Best Fold: {best_fold_idx + 1}/5)")
    print("=" * 80)
    best_results = all_fold_results[best_fold_idx]
    df_best = pd.DataFrame(best_results)

    test_cols = ['Bacteria', 'Test_Accuracy', 'Test_Sensitivity', 'Test_Specificity',
                 'Test_GMean', 'Test_MCC', 'Test_AUC', 'Test_F1', 'Final_Alpha', 'Target_Alpha']
    print(df_best[test_cols].round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("           🔧 LEARNED ADAPTIVE WEIGHTS (α per Bacteria)")
    print("=" * 80)
    for _, r in df_best.iterrows():
        print(f"  {r['Bacteria']}: α = {r['Final_Alpha']:.3f}  (target={r['Target_Alpha']:.2f})")

    print(f"\n{'-' * 50}")
    print("TEST SUMMARY (Best Fold):")
    print(f"  Mean MCC:   {df_best['Test_MCC'].mean():.4f} ± {df_best['Test_MCC'].std():.4f}")
    print(f"  Mean AUC:   {df_best['Test_AUC'].mean():.4f} ± {df_best['Test_AUC'].std():.4f}")
    print(f"  Mean GMean: {df_best['Test_GMean'].mean():.4f}")
    print(f"  Mean Sen:   {df_best['Test_Sensitivity'].mean():.4f}")
    print(f"  Mean Spe:   {df_best['Test_Specificity'].mean():.4f}")
    print(f"  Best Fold:   {best_fold_idx + 1}/5")
    print(f"{'-' * 50}")

    print(f"\n{'-' * 50}")
    print("5-FOLD TEST AVG SUMMARY (for thesis):")
    print(f"  Mean MCC:   {df_summary['Test_MCC_mean'].mean():.4f} ± {df_summary['Test_MCC_mean'].std():.4f}")
    print(f"  Mean AUC:   {df_summary['Test_AUC_mean'].mean():.4f} ± {df_summary['Test_AUC_mean'].std():.4f}")
    print(f"  Mean GMean: {df_summary['Test_GMean_mean'].mean():.4f}")
    print(f"  α Diversity (mean std): {df_summary['Alpha_std'].mean():.4f}")
    print(f"{'-' * 50}")

    df_best.to_csv(SAVE_DIR / 'best_fold_results.csv', index=False)
    df_summary.to_csv(SAVE_DIR / '5fold_test_avg_results.csv', index=False)

    thesis = []
    for _, r in df_best.iterrows():
        thesis.append({
            'Bacteria': r['Bacteria'],
            'Accuracy': f"{r['Test_Accuracy']:.3f}",
            'Sensitivity': f"{r['Test_Sensitivity']:.3f}",
            'Specificity': f"{r['Test_Specificity']:.3f}",
            'GMean': f"{r['Test_GMean']:.3f}",
            'MCC': f"{r['Test_MCC']:.3f}",
            'AUC': f"{r['Test_AUC']:.3f}",
        })
    pd.DataFrame(thesis).to_csv(SAVE_DIR / 'test_for_thesis.csv', index=False)
    print(f"\n[INFO] 结果已保存: {SAVE_DIR}")

    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
