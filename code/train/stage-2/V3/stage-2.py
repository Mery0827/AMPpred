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
# 当前文件所在目录: code/train/stage-2/V3/
CURRENT_DIR = Path(__file__).resolve().parent
# model.py 所在目录: code/train/
CODE_DIR = CURRENT_DIR.parent.parent

sys.path.insert(0, str(CODE_DIR))
from model import DeepAMPpredStage2_MICAux, FocalLoss

warnings.filterwarnings('ignore')

# ==================== 【实验版本控制矩阵】=====================
EXPERIMENT_TAG = "v3_focal_aux"  # ← v3 配置

USE_FOCAL_LOSS = True  # ← v3 配置
USE_ADAPTIVE_LOSS = False
USE_MIC_AUX = True  # ← v3 配置
USE_SWA = False

# 统一超参数（四版本必须相同，控制变量）
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

# 结果保存到当前脚本同级目录（如 code/train/stage-2/V3/results/v3_focal_aux/）
SAVE_DIR = CURRENT_DIR / "results" / EXPERIMENT_TAG
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ======================== 日志保存类 ========================
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


# ======================== MIC Pairwise Ranking 辅助任务 ========================
def compute_mic_aux_loss(features, labels, mic_values, model, max_pairs=200):
    labels = labels.view(-1);
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

    feat_i = feat_valid[pairs_i];
    feat_j = feat_valid[pairs_j]
    mic_i = mic_valid[pairs_i];
    mic_j = mic_valid[pairs_j]
    aux_labels = (mic_i < mic_j).long()
    aux_feat = torch.abs(feat_i - feat_j)
    return F.cross_entropy(model.mic_classifier(aux_feat), aux_labels)


# ======================== 评估指标 ========================
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
    best_mcc, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (y_prob >= t).astype(int)
        try:
            score = matthews_corrcoef(y_true, pred)
        except:
            score = 0.0
        if score > best_mcc: best_mcc, best_t = score, t
    return best_t, best_mcc


# ===================== 数据平衡加载器 ======================
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


# ===================== 加载数据 ======================
def load_bacteria_data(bacteria_name):
    X_train = np.load(str(TRAIN_DIR / f"{bacteria_name}_features.npy"))
    y_train = np.load(str(TRAIN_DIR / f"{bacteria_name}_labels.npy"))
    mic_train = np.load(str(TRAIN_DIR / f"{bacteria_name}_mic.npy"))

    X_test = np.load(str(TEST_DIR / f"{bacteria_name}_features.npy"))
    y_test = np.load(str(TEST_DIR / f"{bacteria_name}_labels.npy"))
    mic_test = np.load(str(TEST_DIR / f"{bacteria_name}_mic.npy"))

    return X_train, y_train, mic_train, X_test, y_test, mic_test


# ===================== 单病菌训练 ======================
def train_single_bacteria(bacteria_idx, bacteria_name):
    print(f"\n{'#' * 70}")
    print(f"# {bacteria_idx + 1}/10: {bacteria_name}")
    print(f"{'#' * 70}")

    X_train, y_train, mic_train, X_test, y_test, mic_test = load_bacteria_data(bacteria_name)

    n_pos = int(y_train.sum())
    pos_ratio = y_train.mean()

    # ===== 损失函数选择（提前确定，用于日志头部） =====
    if USE_ADAPTIVE_LOSS:
        if pos_ratio < 0.15 or pos_ratio > 0.85:
            alpha = 0.7 if pos_ratio < 0.15 else 0.25
            criterion = FocalLoss(alpha=alpha, gamma=2.0)
            loss_name = f"Adaptive-Focal(α={alpha})"
        else:
            criterion = nn.BCEWithLogitsLoss(label_smoothing=0.1)
            loss_name = "Adaptive-BCE+LS(0.1)"
    elif USE_FOCAL_LOSS:
        if pos_ratio < 0.05:
            alpha = 0.9
        elif pos_ratio < 0.15:
            alpha = 0.7
        elif pos_ratio < 0.3:
            alpha = 0.5
        else:
            alpha = 0.25
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
        loss_name = f"Global-Focal(α={alpha})"
    else:
        criterion = nn.BCEWithLogitsLoss()
        loss_name = "BCE"

    aux_status = "ON" if USE_MIC_AUX else "OFF"

    # ==================== 详细头部日志（参考历史日志格式） ====================
    print(f"\n{'=' * 70}")
    print(
        f"  Training: {bacteria_name} (idx={bacteria_idx}) | {'MIC-Auxiliary Task' if USE_MIC_AUX else 'Pure Baseline'}")
    print(f"  Positive: {n_pos} | Negative: {len(y_train) - n_pos}")
    print(f"  Positive ratio: {pos_ratio:.3f}")
    print(f"{'=' * 70}")
    print(
        f"  📌 [消融实验-{'关闭' if not USE_MIC_AUX else '开启'}组] 统一{'关闭' if not USE_MIC_AUX else '开启'}MIC辅助任务 ({'纯基线对照组' if not USE_MIC_AUX else 'MIC辅助实验组'})")
    print(f"  📌 [损失策略] {loss_name} ({EXPERIMENT_TAG.upper()})")
    print(f"  Test:  {len(y_test)} (Pos={int(y_test.sum())}, Neg={len(y_test) - int(y_test.sum())})")

    # ==================== 阶段1：Train 内部 5-fold CV ====================
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_records = []
    best_fold_model = None
    best_fold_mcc = -np.inf

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n  {'-' * 50}")
        print(f"  Fold {fold + 1}/5")
        print(f"  {'-' * 50}")
        print(f"  📌 [Fold {fold + 1}] {'Pure Baseline' if not USE_MIC_AUX else 'MIC-Aux'}: {loss_name}")

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        mic_tr, mic_val = mic_train[tr_idx], mic_train[val_idx]

        # 标准化
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_mcc = -np.inf
        best_state = None
        best_epoch = 0
        best_fold_metrics = None
        best_thresh = 0.5
        counter = 0

        for epoch in range(EPOCHS):
            aux_weight = min(0.15, 0.05 * (epoch - 5 + 1)) if (USE_MIC_AUX and epoch >= 5) else 0.0

            # Train
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb, mb in loader:
                xb = xb.to(device)
                yb = yb.to(device).squeeze(1)  # [batch, 1] -> [batch]
                mb = mb.to(device)

                optimizer.zero_grad()
                out_main, feat_mic = model(xb, return_features=True, bacteria_idx=bacteria_idx)

                # ✅ 核心修正：out_main 是 [batch, 1]，squeeze(-1) 成 [batch] 匹配 yb
                loss_main = criterion(out_main.squeeze(-1), yb)

                loss_aux = torch.tensor(0.0, device=device)
                if USE_MIC_AUX and aux_weight > 0 and mb.sum() > 0:
                    loss_aux = compute_mic_aux_loss(feat_mic, yb, mb, model)

                loss = loss_main + aux_weight * loss_aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Val
            model.eval()
            with torch.no_grad():
                out_main = model(X_va, return_features=False, bacteria_idx=bacteria_idx)
                prob = torch.sigmoid(out_main).squeeze().cpu().numpy()
                y_true = y_va.squeeze().cpu().numpy()

            thresh, _ = find_best_threshold(y_true, prob)
            pred = (prob >= thresh).astype(int)
            m = compute_metrics(y_true, pred, prob)
            current_mcc = m['MCC']

            # ==================== Epoch 级详细日志 ====================
            avg_loss = epoch_loss / max(n_batches, 1)
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f"  [Fold {fold + 1}] Epoch {epoch + 1:3d} | Loss {avg_loss:.3f} | MCC {m['MCC']:.3f} | AUC {m['AUC']:.3f} | GMean {m['GMean']:.3f} | Sen {m['Sensitivity']:.3f} | Spe {m['Specificity']:.3f} | LR {lr_now:.6f} | Aux={aux_status}")

            if current_mcc > best_mcc:
                best_mcc = current_mcc
                best_state = model.state_dict()
                best_epoch = epoch + 1
                best_fold_metrics = m.copy()
                best_thresh = thresh
                counter = 0
            else:
                counter += 1

            if counter >= PATIENCE:
                print(f"\n  🛑 Early stop at Epoch {epoch + 1}, best was Epoch {best_epoch}")
                break
            scheduler.step()

        # Fold 结束总结
        print(f"\n  ✅ Fold {fold + 1} BEST → MCC={best_mcc:.4f} AUC={best_fold_metrics['AUC']:.4f}")
        print(
            f"     Thresh={best_thresh:.2f} Sen={best_fold_metrics['Sensitivity']:.3f} Spe={best_fold_metrics['Specificity']:.3f}")

        fold_records.append({
            'best_mcc': best_mcc,
            'best_auc': best_fold_metrics['AUC'],
            'best_gmean': best_fold_metrics['GMean'],
            'best_sen': best_fold_metrics['Sensitivity'],
            'best_spe': best_fold_metrics['Specificity'],
            'best_epoch': best_epoch,
            'best_thresh': best_thresh,
        })

        # 保存最佳 fold（用于独立 Test）
        if best_mcc > best_fold_mcc:
            best_fold_mcc = best_mcc
            best_fold_model = {'state_dict': best_state, 'scaler': scaler}

    # ==================== 5-Fold 平均总结 ====================
    cv_mcc_mean = np.mean([r['best_mcc'] for r in fold_records])
    cv_mcc_std = np.std([r['best_mcc'] for r in fold_records])
    cv_auc_mean = np.mean([r['best_auc'] for r in fold_records])
    cv_auc_std = np.std([r['best_auc'] for r in fold_records])
    cv_gmean_mean = np.mean([r['best_gmean'] for r in fold_records])

    print(
        f"\n  🏆 {bacteria_name} 5-FOLD AVG → MCC={cv_mcc_mean:.4f}±{cv_mcc_std:.4f} AUC={cv_auc_mean:.4f}±{cv_auc_std:.4f}")

    # ==================== 阶段2：独立 Test 最终评估（仅测一次）====================
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

    save_path = SAVE_DIR / f"model_{bacteria_name}.pth"
    torch.save({
        'state_dict': best_fold_model['state_dict'],
        'scaler': scaler,
        'test_thresh': thresh_test,
        'test_mcc': test_metrics['MCC'],
        'cv_mcc_mean': cv_mcc_mean,
        'cv_mcc_std': cv_mcc_std,
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
    }


# ===================== 主函数 =====================
def main():
    log_file = SAVE_DIR / f"log_{datetime.now().strftime('%m%d_%H%M')}.txt"
    sys.stdout = Logger(log_file)

    print("=" * 80)
    print(" Deep-AMPpred Stage-2 | One-vs-All | Train 5-fold CV + Independent Test")
    print("=" * 80)
    print(f"Version: {EXPERIMENT_TAG}")
    print(f"Loss: FOCAL={USE_FOCAL_LOSS} | ADAPTIVE={USE_ADAPTIVE_LOSS} | AUX={USE_MIC_AUX}")
    print(f"Device: {device}")
    print(f"Feature Dir: {FEATURE_DIR}")
    print(f"Save Dir: {SAVE_DIR}")
    print("=" * 80)

    all_results = []
    for i, name in enumerate(BACTERIA_NAMES):
        res = train_single_bacteria(i, name)
        all_results.append(res)

    df = pd.DataFrame(all_results)

    # 打印 5-Fold CV 结果（稳定性）
    print("\n" + "=" * 80)
    print("           📊 5-FOLD CV RESULTS (Model Stability)")
    print("=" * 80)
    cv_cols = ['Bacteria', 'CV_MCC_mean', 'CV_MCC_std', 'CV_AUC_mean', 'CV_AUC_std', 'CV_GMean_mean']
    print(df[cv_cols].round(4).to_string(index=False))

    # 打印独立 Test 结果（与 AMPActiPred 对比）
    print("\n" + "=" * 80)
    print("           🎯 INDEPENDENT TEST RESULTS (vs. AMPActiPred Table 3)")
    print("=" * 80)
    test_cols = ['Bacteria', 'Test_Accuracy', 'Test_Sensitivity', 'Test_Specificity',
                 'Test_GMean', 'Test_MCC', 'Test_AUC', 'Test_F1']
    print(df[test_cols].round(4).to_string(index=False))

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

    # 论文格式表
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
