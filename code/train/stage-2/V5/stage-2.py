import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
import warnings
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

# ==================== 路径配置（加入 sys.path 以导入 model.py）=====================
# 当前文件所在目录: code/train/stage-2/V5/
CURRENT_DIR = Path(__file__).resolve().parent
# model.py 所在目录: code/train/
CODE_DIR = CURRENT_DIR.parent.parent

sys.path.insert(0, str(CODE_DIR))
from model import DeepAMPpredStage2_MICAux, AdaptiveBCEFocalLoss

warnings.filterwarnings('ignore')

EXPERIMENT_TAG = "v5_prior_anchored"

USE_LEARNABLE_ADAPTIVE = True
USE_MIC_AUX = False

BATCH_SIZE = 32
EPOCHS = 150
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20
DROPOUT = 0.3
LSTM_HIDDEN = 64
LSTM_LAYERS = 2

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

# 结果保存到当前脚本同级目录（如 code/train/stage-2/V5/results/v5_prior_anchored/）
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


def load_bacteria_data(bacteria_name):
    X_train = np.load(str(TRAIN_DIR / f"{bacteria_name}_features.npy"))
    y_train = np.load(str(TRAIN_DIR / f"{bacteria_name}_labels.npy"))
    mic_train = np.load(str(TRAIN_DIR / f"{bacteria_name}_mic.npy"))

    X_test = np.load(str(TEST_DIR / f"{bacteria_name}_features.npy"))
    y_test = np.load(str(TEST_DIR / f"{bacteria_name}_labels.npy"))
    mic_test = np.load(str(TEST_DIR / f"{bacteria_name}_mic.npy"))

    return X_train, y_train, mic_train, X_test, y_test, mic_test


# ===================== 基于不平衡度的先验配置（V6逻辑移植）=====================
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


def train_single_bacteria(bacteria_idx, bacteria_name):
    print(f"\n{'#' * 70}")
    print(f"# {bacteria_idx + 1}/10: {bacteria_name} | Prior-Anchored Single-Task")
    print(f"{'#' * 70}")

    X_train, y_train, mic_train, X_test, y_test, mic_test = load_bacteria_data(bacteria_name)

    n_pos = int(y_train.sum())
    pos_ratio = y_train.mean()

    # 获取先验配置
    cfg = get_task_alpha_config(pos_ratio)

    print(f"\n{'=' * 70}")
    print(f"  Training: {bacteria_name} (idx={bacteria_idx}) | Pure Baseline (No Aux)")
    print(f"  Positive: {n_pos} | Negative: {len(y_train) - n_pos}")
    print(f"  Positive ratio: {pos_ratio:.3f}")
    print(
        f"  Alpha Config: target={cfg['target_alpha']:.2f} | init_logit={cfg['init_logit']:+.2f} | reg={cfg['reg_weight']:.1f}")
    print(f"{'=' * 70}")
    print(f"  Test:  {len(y_test)} (Pos={int(y_test.sum())}, Neg={len(y_test) - int(y_test.sum())})")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_records = []
    best_fold_model = None
    best_fold_mcc = -np.inf

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n  {'-' * 50}")
        print(f"  Fold {fold + 1}/5")
        print(f"  {'-' * 50}")

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        mic_tr, mic_val = mic_train[tr_idx], mic_train[val_idx]

        N_tr, seq_len, feat_dim = X_tr.shape
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(-1, feat_dim)).reshape(N_tr, seq_len, feat_dim)
        N_va, _, _ = X_val.shape
        X_val = scaler.transform(X_val.reshape(-1, feat_dim)).reshape(N_va, seq_len, feat_dim)

        loader = create_balanced_loader(X_tr, y_tr, mic_tr, batch_size=BATCH_SIZE)
        X_va = torch.FloatTensor(X_val).to(device)
        y_va = torch.FloatTensor(y_val).to(device).unsqueeze(1)

        torch.manual_seed(42 + fold)
        model = DeepAMPpredStage2_MICAux(
            input_dim=480, seq_len=100, num_bacteria=NUM_BACTERIA,
            lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=DROPOUT
        ).to(device)

        # ✅ V5-Prior核心：带先验锚定的可学习损失
        criterion = AdaptiveBCEFocalLoss(
            gamma=1.5,
            focal_alpha=0.25,
            reg_weight=cfg['reg_weight'],
            smoothing=0.1,
            init_logit=cfg['init_logit'],
            target_alpha=cfg['target_alpha']
        ).to(device)

        # 主网络优化器
        optimizer_model = torch.optim.AdamW(
            model.parameters(),
            lr=LR, weight_decay=WEIGHT_DECAY
        )
        # α 独立优化器（V6逻辑）
        optimizer_alpha = torch.optim.Adam(
            [criterion.logit_alpha],
            lr=cfg['alpha_lr']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_model, T_0=10, T_mult=2
        )

        best_mcc = -np.inf
        best_state = None
        best_epoch = 0
        best_fold_metrics = None
        best_thresh = 0.5
        best_alpha = 0.55
        counter = 0

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb, mb in loader:
                xb = xb.to(device)
                yb = yb.to(device).squeeze(1)
                mb = mb.to(device)

                optimizer_model.zero_grad()
                optimizer_alpha.zero_grad()

                out_main = model(xb, return_features=False, bacteria_idx=bacteria_idx)
                loss = criterion(out_main.squeeze(-1), yb)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_model.step()
                optimizer_alpha.step()

                epoch_loss += loss.item()
                n_batches += 1

            model.eval()
            with torch.no_grad():
                out_main = model(X_va, return_features=False, bacteria_idx=bacteria_idx)
                prob = torch.sigmoid(out_main).squeeze().cpu().numpy()
                y_true = y_va.squeeze().cpu().numpy()

            thresh, _ = find_best_threshold(y_true, prob)
            pred = (prob >= thresh).astype(int)
            m = compute_metrics(y_true, pred, prob)
            current_mcc = m['MCC']

            avg_loss = epoch_loss / max(n_batches, 1)
            lr_now = optimizer_model.param_groups[0]['lr']

            alpha_str = ""
            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                alpha_str = f"| α={criterion.alpha.item():.3f}"

            print(
                f"  [Fold {fold + 1}] Epoch {epoch + 1:3d} | Loss {avg_loss:.3f} | MCC {m['MCC']:.3f} | AUC {m['AUC']:.3f} | GMean {m['GMean']:.3f} | LR {lr_now:.6f} {alpha_str}")

            if current_mcc > best_mcc:
                best_mcc = current_mcc
                best_state = {
                    'model': model.state_dict(),
                    'criterion': criterion.state_dict(),
                    'alpha': criterion.alpha.item()
                }
                best_epoch = epoch + 1
                best_fold_metrics = m.copy()
                best_thresh = thresh
                best_alpha = criterion.alpha.item()
                counter = 0
            else:
                counter += 1

            if counter >= PATIENCE:
                print(f"\n  🛑 Early stop at Epoch {epoch + 1}, best was Epoch {best_epoch}")
                break

            scheduler.step()

        print(f"\n  ✅ Fold {fold + 1} BEST → MCC={best_mcc:.4f} AUC={best_fold_metrics['AUC']:.4f}")
        print(
            f"     Thresh={best_thresh:.2f} Sen={best_fold_metrics['Sensitivity']:.3f} Spe={best_fold_metrics['Specificity']:.3f}")
        print(f"     Best α={best_alpha:.4f} (target={cfg['target_alpha']:.2f})")

        fold_records.append({
            'best_mcc': best_mcc,
            'best_auc': best_fold_metrics['AUC'],
            'best_gmean': best_fold_metrics['GMean'],
            'best_sen': best_fold_metrics['Sensitivity'],
            'best_spe': best_fold_metrics['Specificity'],
            'best_epoch': best_epoch,
            'best_thresh': best_thresh,
            'fold_alpha': best_alpha,
        })

        if best_mcc > best_fold_mcc:
            best_fold_mcc = best_mcc
            best_fold_model = {
                'state_dict': best_state['model'],
                'criterion_state': best_state['criterion'],
                'scaler': scaler,
                'alpha': best_state['alpha']
            }

    cv_mcc_mean = np.mean([r['best_mcc'] for r in fold_records])
    cv_mcc_std = np.std([r['best_mcc'] for r in fold_records])
    cv_auc_mean = np.mean([r['best_auc'] for r in fold_records])
    cv_auc_std = np.std([r['best_auc'] for r in fold_records])
    cv_gmean_mean = np.mean([r['best_gmean'] for r in fold_records])

    print(
        f"\n  🏆 {bacteria_name} 5-FOLD AVG → MCC={cv_mcc_mean:.4f}±{cv_mcc_std:.4f} AUC={cv_auc_mean:.4f}±{cv_auc_std:.4f}")

    alphas = [r['fold_alpha'] for r in fold_records]
    print(
        f"     Learned α per fold: {[f'{a:.3f}' for a in alphas]} (mean={np.mean(alphas):.3f}, target={cfg['target_alpha']:.2f})")

    scaler = best_fold_model['scaler']
    N_te, seq_len, feat_dim = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, feat_dim)).reshape(N_te, seq_len, feat_dim)

    X_te_t = torch.FloatTensor(X_test_scaled).to(device)
    y_te_t = torch.FloatTensor(y_test).to(device)

    final_model = DeepAMPpredStage2_MICAux(
        input_dim=480, seq_len=100, num_bacteria=NUM_BACTERIA,
        lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=DROPOUT
    ).to(device)
    final_model.load_state_dict(best_fold_model['state_dict'])
    final_model.eval()

    with torch.no_grad():
        out_main = final_model(X_te_t, return_features=False, bacteria_idx=bacteria_idx)
        prob_test = torch.sigmoid(out_main).squeeze().cpu().numpy()
        y_true_test = y_te_t.cpu().numpy()

    thresh_test, _ = find_best_threshold(y_true_test, prob_test)
    pred_test = (prob_test >= thresh_test).astype(int)
    test_metrics = compute_metrics(y_true_test, pred_test, prob_test)

    print(f"\n  🎯 Independent TEST: MCC={test_metrics['MCC']:.4f} AUC={test_metrics['AUC']:.4f} "
          f"GMean={test_metrics['GMean']:.4f}")

    final_alpha = best_fold_model.get('alpha', None)

    save_path = SAVE_DIR / f"model_{bacteria_name}.pth"
    torch.save({
        'state_dict': best_fold_model['state_dict'],
        'criterion_state': best_fold_model.get('criterion_state'),
        'scaler': scaler,
        'test_thresh': thresh_test,
        'test_mcc': test_metrics['MCC'],
        'cv_mcc_mean': cv_mcc_mean,
        'cv_mcc_std': cv_mcc_std,
        'final_alpha': final_alpha,
    }, save_path)
    print(f"  💾 模型已保存: {save_path}")

    return {
        'Bacteria': bacteria_name,
        'CV_MCC_mean': cv_mcc_mean,
        'CV_MCC_std': cv_mcc_std,
        'CV_AUC_mean': cv_auc_mean,
        'CV_AUC_std': cv_auc_std,
        'CV_GMean_mean': cv_gmean_mean,
        'Test_MCC': test_metrics['MCC'],
        'Test_AUC': test_metrics['AUC'],
        'Test_GMean': test_metrics['GMean'],
        'Test_Sensitivity': test_metrics['Sensitivity'],
        'Test_Specificity': test_metrics['Specificity'],
        'Test_Accuracy': test_metrics['Accuracy'],
        'Test_F1': test_metrics['F1'],
        'Final_Alpha': final_alpha,
        'Target_Alpha': cfg['target_alpha'],
    }


def main():
    log_file = SAVE_DIR / f"log_{datetime.now().strftime('%m%d_%H%M')}.txt"
    sys.stdout = Logger(log_file)

    print("=" * 80)
    print(" Deep-AMPpred Stage-2 | V5-Prior: Single-Task + Prior-Anchored α")
    print("=" * 80)
    print(f"Version: {EXPERIMENT_TAG}")
    print(f"Loss: LEARNABLE_ADAPTIVE={USE_LEARNABLE_ADAPTIVE} | AUX={USE_MIC_AUX}")
    print(f"Device: {device}")
    print(f"Feature Dir: {FEATURE_DIR}")
    print(f"Save Dir: {SAVE_DIR}")
    print("=" * 80)

    all_results = []
    for i, name in enumerate(BACTERIA_NAMES):
        res = train_single_bacteria(i, name)
        all_results.append(res)

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("           📊 5-FOLD CV RESULTS")
    print("=" * 80)
    cv_cols = ['Bacteria', 'CV_MCC_mean', 'CV_MCC_std', 'CV_AUC_mean', 'CV_AUC_std', 'CV_GMean_mean']
    print(df[cv_cols].round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("           🎯 INDEPENDENT TEST RESULTS")
    print("=" * 80)
    test_cols = ['Bacteria', 'Test_Accuracy', 'Test_Sensitivity', 'Test_Specificity',
                 'Test_GMean', 'Test_MCC', 'Test_AUC', 'Test_F1']
    print(df[test_cols].round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("           🔧 LEARNED ADAPTIVE WEIGHTS (α per Bacteria)")
    print("=" * 80)
    for _, r in df.iterrows():
        alpha_val = r['Final_Alpha']
        alpha_str = f"{alpha_val:.3f}" if alpha_val is not None else "N/A"
        print(f"  {r['Bacteria']}: α = {alpha_str}  (target={r['Target_Alpha']:.2f})")

    print(f"\n{'-' * 50}")
    print("TEST SUMMARY (for thesis):")
    print(f"  Mean MCC:   {df['Test_MCC'].mean():.4f} ± {df['Test_MCC'].std():.4f}")
    print(f"  Mean AUC:   {df['Test_AUC'].mean():.4f} ± {df['Test_AUC'].std():.4f}")
    print(f"  Mean GMean: {df['Test_GMean'].mean():.4f}")
    print(f"  Mean Sen:   {df['Test_Sensitivity'].mean():.4f}")
    print(f"  Mean Spe:   {df['Test_Specificity'].mean():.4f}")
    print(f"{'-' * 50}")

    results_csv = SAVE_DIR / 'all_results.csv'
    df.to_csv(results_csv, index=False)
    print(f"\n[INFO] 结果已保存: {results_csv}")

    thesis = []
    for _, r in df.iterrows():
        thesis.append({
            'Bacteria': r['Bacteria'],
            'Accuracy': f"{r['Test_Accuracy']:.3f}",
            'Sensitivity': f"{r['Test_Sensitivity']:.3f}",
            'Specificity': f"{r['Test_Specificity']:.3f}",
            'GMean': f"{r['Test_GMean']:.3f}",
            'MCC': f"{r['Test_MCC']:.3f}",
            'AUC': f"{r['Test_AUC']:.3f}",
        })
    thesis_csv = SAVE_DIR / 'test_for_thesis.csv'
    pd.DataFrame(thesis).to_csv(thesis_csv, index=False)
    print(f"[INFO] 论文表格已保存: {thesis_csv}")

    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
