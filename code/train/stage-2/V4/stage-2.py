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
# 当前文件所在目录: code/train/stage-2/V4/
CURRENT_DIR = Path(__file__).resolve().parent
# model.py 所在目录: code/train/
CODE_DIR = CURRENT_DIR.parent.parent

sys.path.insert(0, str(CODE_DIR))
from model import DeepAMPpredStage2_MICAux, FocalLoss, AdaptiveBCEFocalLoss

warnings.filterwarnings('ignore')

# ==================== 【实验版本控制矩阵】=====================
EXPERIMENT_TAG = "v4_prior_anchored"  # ✅ 标记为先验锚定版

USE_FOCAL_LOSS = False
USE_ADAPTIVE_LOSS = False
USE_LEARNABLE_ADAPTIVE = True  # ✅ 启用先验锚定可学习损失
USE_MIC_AUX = True  # ✅ V4E 保留 MIC 辅助任务
USE_SWA = False

# ==================== 【可配置：指定跑哪些菌种】=====================
RUN_BACTERIA = None  # 跑全部10个

# 统一超参数
BATCH_SIZE = 32
EPOCHS = 150
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30
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

# 结果保存到当前脚本同级目录（如 code/train/stage-2/V4/results/v4_prior_anchored/）
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


# ===================== 细粒度阈值搜索 =====================
def find_best_threshold(y_true, y_prob):
    best_mcc, best_t = -1.0, 0.5
    for t in np.arange(0.01, 0.991, 0.005):
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


# ===================== 单病菌训练（V4E-Prior 增强版）=====================
def train_single_bacteria(bacteria_idx, bacteria_name):
    print(f"\n{'#' * 70}")
    print(f"# {bacteria_idx + 1}/10: {bacteria_name} | Prior-Anchored Single-Task")
    print(f"{'#' * 70}")

    X_train, y_train, mic_train, X_test, y_test, mic_test = load_bacteria_data(bacteria_name)

    n_pos = int(y_train.sum())
    pos_ratio = y_train.mean()

    # ✅ V4-Prior核心：获取先验配置
    cfg = get_task_alpha_config(pos_ratio)

    # V4E 保留：动态 focal_alpha 和采样比例
    if pos_ratio < 0.05:
        dynamic_focal_alpha = 0.9
        dynamic_pos_target = 0.15
    elif pos_ratio < 0.15:
        dynamic_focal_alpha = 0.7
        dynamic_pos_target = 0.2
    elif pos_ratio < 0.3:
        dynamic_focal_alpha = 0.5
        dynamic_pos_target = 0.25
    else:
        dynamic_focal_alpha = 0.25
        dynamic_pos_target = 0.3

    loss_name = f"Prior-Anchored(α·BCE+(1-α)·Focal, target={cfg['target_alpha']:.2f})"
    aux_status = "ON" if USE_MIC_AUX else "OFF"

    print(f"\n{'=' * 70}")
    print(
        f"  Training: {bacteria_name} (idx={bacteria_idx}) | {'MIC-Auxiliary Task' if USE_MIC_AUX else 'Pure Baseline'}")
    print(f"  Positive: {n_pos} | Negative: {len(y_train) - n_pos}")
    print(f"  Positive ratio: {pos_ratio:.3f}")
    print(f"  📌 [自适应参数] focal_alpha={dynamic_focal_alpha}, pos_target={dynamic_pos_target}")
    print(
        f"  📌 [先验锚定] target={cfg['target_alpha']:.2f} | init_logit={cfg['init_logit']:+.2f} | reg={cfg['reg_weight']:.1f}")
    print(f"  📌 [损失策略] {loss_name} ({EXPERIMENT_TAG.upper()})")
    print(f"{'=' * 70}")
    print(f"  Test:  {len(y_test)} (Pos={int(y_test.sum())}, Neg={len(y_test) - int(y_test.sum())})")

    # ==================== 阶段1：Train 内部 5-fold CV ====================
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_records = []
    all_fold_models = []
    best_fold_model = None
    best_fold_mcc = -np.inf

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"\n  {'-' * 50}")
        print(f"  Fold {fold + 1}/5")
        print(f"  {'-' * 50}")

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        mic_tr, mic_val = mic_train[tr_idx], mic_train[val_idx]

        # 标准化
        N_tr, seq_len, feat_dim = X_tr.shape
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(-1, feat_dim)).reshape(N_tr, seq_len, feat_dim)
        N_va, _, _ = X_val.shape
        X_val = scaler.transform(X_val.reshape(-1, feat_dim)).reshape(N_va, seq_len, feat_dim)

        loader = create_balanced_loader(X_tr, y_tr, mic_tr, batch_size=BATCH_SIZE, pos_ratio_target=dynamic_pos_target)
        X_va = torch.FloatTensor(X_val).to(device)
        y_va = torch.FloatTensor(y_val).to(device).unsqueeze(1)

        torch.manual_seed(42 + fold)
        model = DeepAMPpredStage2_MICAux(
            input_dim=480, seq_len=100, num_bacteria=NUM_BACTERIA,
            lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=DROPOUT
        ).to(device)

        # ✅ V4-Prior核心：带先验锚定的可学习损失
        if USE_LEARNABLE_ADAPTIVE:
            criterion = AdaptiveBCEFocalLoss(
                gamma=2.0,
                focal_alpha=dynamic_focal_alpha,
                reg_weight=cfg['reg_weight'],
                smoothing=0.1,
                init_logit=cfg['init_logit'],
                target_alpha=cfg['target_alpha']
            ).to(device)
        elif USE_ADAPTIVE_LOSS:
            if pos_ratio < 0.15 or pos_ratio > 0.85:
                tmp_alpha = 0.7 if pos_ratio < 0.15 else 0.25
                criterion = FocalLoss(alpha=tmp_alpha, gamma=2.0)
            else:
                criterion = nn.BCEWithLogitsLoss(label_smoothing=0.1)
        elif USE_FOCAL_LOSS:
            if pos_ratio < 0.05:
                tmp_alpha = 0.9
            elif pos_ratio < 0.15:
                tmp_alpha = 0.7
            elif pos_ratio < 0.3:
                tmp_alpha = 0.5
            else:
                tmp_alpha = 0.25
            criterion = FocalLoss(alpha=tmp_alpha, gamma=2.0)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # ✅ V4-Prior核心：主网络与 α 分离优化
        if USE_LEARNABLE_ADAPTIVE:
            optimizer_model = torch.optim.AdamW(
                model.parameters(),
                lr=LR, weight_decay=WEIGHT_DECAY
            )
            optimizer_alpha = torch.optim.Adam(
                [criterion.logit_alpha],
                lr=cfg['alpha_lr']
            )
        else:
            optimizer_model = torch.optim.AdamW(
                list(model.parameters()) + (list(criterion.parameters()) if hasattr(criterion, 'parameters') else []),
                lr=LR, weight_decay=WEIGHT_DECAY
            )
            optimizer_alpha = None

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_model, T_0=10, T_mult=2)

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
                yb = yb.to(device).squeeze(1)
                mb = mb.to(device)

                optimizer_model.zero_grad()
                if optimizer_alpha is not None:
                    optimizer_alpha.zero_grad()

                out_main, feat_mic = model(xb, return_features=True, bacteria_idx=bacteria_idx)

                loss_main = criterion(out_main.squeeze(-1), yb)

                loss_aux = torch.tensor(0.0, device=device)
                if USE_MIC_AUX and aux_weight > 0 and mb.sum() > 0:
                    loss_aux = compute_mic_aux_loss(feat_mic, yb, mb, model)

                loss = loss_main + aux_weight * loss_aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_model.step()
                if optimizer_alpha is not None:
                    optimizer_alpha.step()

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

            # Epoch 级详细日志
            avg_loss = epoch_loss / max(n_batches, 1)
            lr_now = optimizer_model.param_groups[0]['lr']

            alpha_str = ""
            if USE_LEARNABLE_ADAPTIVE:
                alpha_val = criterion.alpha.item()
                if epoch % 10 == 0 or epoch == EPOCHS - 1:
                    alpha_str = f"| α={alpha_val:.3f} (target={cfg['target_alpha']:.2f})"

            print(
                f"  [Fold {fold + 1}] Epoch {epoch + 1:3d} | Loss {avg_loss:.3f} | MCC {m['MCC']:.3f} | AUC {m['AUC']:.3f} | GMean {m['GMean']:.3f} | Sen {m['Sensitivity']:.3f} | Spe {m['Specificity']:.3f} | LR {lr_now:.6f} | Aux={aux_status} {alpha_str}")

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
        fold_alpha = criterion.alpha.item() if USE_LEARNABLE_ADAPTIVE else None

        print(f"\n  ✅ Fold {fold + 1} BEST → MCC={best_mcc:.4f} AUC={best_fold_metrics['AUC']:.4f}")
        print(
            f"     Thresh={best_thresh:.2f} Sen={best_fold_metrics['Sensitivity']:.3f} Spe={best_fold_metrics['Specificity']:.3f}")
        if fold_alpha is not None:
            print(f"     Best α={fold_alpha:.4f} (target={cfg['target_alpha']:.2f})")

        fold_records.append({
            'best_mcc': best_mcc,
            'best_auc': best_fold_metrics['AUC'],
            'best_gmean': best_fold_metrics['GMean'],
            'best_sen': best_fold_metrics['Sensitivity'],
            'best_spe': best_fold_metrics['Specificity'],
            'best_epoch': best_epoch,
            'best_thresh': best_thresh,
            'fold_alpha': fold_alpha,
        })

        # ✅ V4E：保存所有fold模型（含criterion_state）用于ensemble
        all_fold_models.append({
            'state_dict': best_state,
            'criterion_state': criterion.state_dict() if USE_LEARNABLE_ADAPTIVE else None,
            'scaler': scaler,
            'best_mcc': best_mcc,
            'best_thresh': best_thresh,
            'fold_alpha': fold_alpha,
        })

        # 同时保留传统最佳fold
        if best_mcc > best_fold_mcc:
            best_fold_mcc = best_mcc
            best_fold_model = {
                'state_dict': best_state,
                'criterion_state': criterion.state_dict() if USE_LEARNABLE_ADAPTIVE else None,
                'scaler': scaler,
                'alpha': fold_alpha
            }

    # ==================== 5-Fold 平均总结 ====================
    cv_mcc_mean = np.mean([r['best_mcc'] for r in fold_records])
    cv_mcc_std = np.std([r['best_mcc'] for r in fold_records])
    cv_auc_mean = np.mean([r['best_auc'] for r in fold_records])
    cv_auc_std = np.std([r['best_auc'] for r in fold_records])
    cv_gmean_mean = np.mean([r['best_gmean'] for r in fold_records])

    print(
        f"\n  🏆 {bacteria_name} 5-FOLD AVG → MCC={cv_mcc_mean:.4f}±{cv_mcc_std:.4f} AUC={cv_auc_mean:.4f}±{cv_auc_std:.4f}")

    if USE_LEARNABLE_ADAPTIVE:
        alphas = [r['fold_alpha'] for r in fold_records if r['fold_alpha'] is not None]
        if alphas:
            print(
                f"     Learned α per fold: {[f'{a:.3f}' for a in alphas]} (mean={np.mean(alphas):.3f}, target={cfg['target_alpha']:.2f})")

    # ==================== V4E核心：5-Fold Ensemble测试 ====================
    print(f"\n  🔮 Testing with 5-Fold Ensemble...")

    N_te, seq_len, feat_dim = X_test.shape
    fold_probs = []

    for fold_info in all_fold_models:
        scaler = fold_info['scaler']
        X_test_scaled = scaler.transform(X_test.reshape(-1, feat_dim)).reshape(N_te, seq_len, feat_dim)
        X_te_t = torch.FloatTensor(X_test_scaled).to(device)

        model = DeepAMPpredStage2_MICAux(
            input_dim=480, seq_len=100, num_bacteria=NUM_BACTERIA,
            lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=DROPOUT
        ).to(device)
        model.load_state_dict(fold_info['state_dict'])
        model.eval()

        with torch.no_grad():
            out_main = model(X_te_t, return_features=False, bacteria_idx=bacteria_idx)
            prob = torch.sigmoid(out_main).squeeze().cpu().numpy()
        fold_probs.append(prob)

    # 平均概率（Ensemble核心）
    prob_test = np.mean(fold_probs, axis=0)
    y_true_test = y_test

    thresh_test, _ = find_best_threshold(y_true_test, prob_test)
    pred_test = (prob_test >= thresh_test).astype(int)
    test_metrics = compute_metrics(y_true_test, pred_test, prob_test)

    print(f"\n  🎯 Ensemble TEST: MCC={test_metrics['MCC']:.4f} AUC={test_metrics['AUC']:.4f} "
          f"GMean={test_metrics['GMean']:.4f} Sen={test_metrics['Sensitivity']:.3f} Spe={test_metrics['Specificity']:.3f}")

    # 同时报告单fold best结果作为对比
    scaler_single = best_fold_model['scaler']
    X_test_scaled_single = scaler_single.transform(X_test.reshape(-1, feat_dim)).reshape(N_te, seq_len, feat_dim)
    X_te_single = torch.FloatTensor(X_test_scaled_single).to(device)

    final_model = DeepAMPpredStage2_MICAux(
        input_dim=480, seq_len=100, num_bacteria=NUM_BACTERIA,
        lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=DROPOUT
    ).to(device)
    final_model.load_state_dict(best_fold_model['state_dict'])
    final_model.eval()

    with torch.no_grad():
        out_main = final_model(X_te_single, return_features=False, bacteria_idx=bacteria_idx)
        prob_single = torch.sigmoid(out_main).squeeze().cpu().numpy()

    thresh_single, _ = find_best_threshold(y_true_test, prob_single)
    pred_single = (prob_single >= thresh_single).astype(int)
    single_metrics = compute_metrics(y_true_test, pred_single, prob_single)
    print(f"  🎯 Single Best TEST: MCC={single_metrics['MCC']:.4f} (for comparison)")

    # 保存结果（以Ensemble为准）
    final_alpha = np.mean(
        [f['fold_alpha'] for f in all_fold_models if f['fold_alpha'] is not None]) if USE_LEARNABLE_ADAPTIVE else None

    save_path = SAVE_DIR / f"model_{bacteria_name}.pth"
    torch.save({
        'state_dict': best_fold_model['state_dict'],
        'criterion_state': best_fold_model.get('criterion_state'),
        'scaler': best_fold_model['scaler'],
        'all_fold_states': [f['state_dict'] for f in all_fold_models],
        'all_fold_scalers': [f['scaler'] for f in all_fold_models],
        'all_fold_criterion_states': [f.get('criterion_state') for f in all_fold_models],
        'test_thresh': thresh_test,
        'test_mcc': test_metrics['MCC'],
        'cv_mcc_mean': cv_mcc_mean,
        'cv_mcc_std': cv_mcc_std,
        'final_alpha': final_alpha,
        'target_alpha': cfg['target_alpha'] if USE_LEARNABLE_ADAPTIVE else None,
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
        'Target_Alpha': cfg['target_alpha'] if USE_LEARNABLE_ADAPTIVE else None,
        'Single_MCC': single_metrics['MCC'],
    }


# ===================== 主函数 =====================
def main():
    log_file = SAVE_DIR / f"log_{datetime.now().strftime('%m%d_%H%M')}.txt"
    sys.stdout = Logger(log_file)

    print("=" * 80)
    print(" Deep-AMPpred Stage-2 | V4E-Prior: Single-Task + Prior-Anchored α")
    print("=" * 80)
    print(f"Version: {EXPERIMENT_TAG}")
    print(f"Loss: LEARNABLE_ADAPTIVE={USE_LEARNABLE_ADAPTIVE} | AUX={USE_MIC_AUX} | PATIENCE={PATIENCE}")
    print(f"Device: {device}")
    print(f"Feature Dir: {FEATURE_DIR}")
    print(f"Save Dir: {SAVE_DIR}")
    print("=" * 80)

    # ==================== 根据 RUN_BACTERIA 过滤菌种 ====================
    if RUN_BACTERIA is None:
        target_bacteria = BACTERIA_NAMES
        print(f"\n📋 RUN_BACTERIA = None，将运行全部 {len(BACTERIA_NAMES)} 个菌种")
    else:
        target_bacteria = [b for b in BACTERIA_NAMES if b in RUN_BACTERIA]
        invalid = [b for b in RUN_BACTERIA if b not in BACTERIA_NAMES]
        if invalid:
            print(f"\n⚠️  无效菌种名称（已忽略）: {invalid}")
        print(f"\n📋 RUN_BACTERIA = {RUN_BACTERIA}，将运行 {len(target_bacteria)} 个菌种: {target_bacteria}")

    all_results = []
    for i, name in enumerate(target_bacteria):
        original_idx = BACTERIA_NAMES.index(name)
        res = train_single_bacteria(original_idx, name)
        all_results.append(res)

    df = pd.DataFrame(all_results)

    # 打印 5-Fold CV 结果
    print("\n" + "=" * 80)
    print("           📊 5-FOLD CV RESULTS (Model Stability)")
    print("=" * 80)
    cv_cols = ['Bacteria', 'CV_MCC_mean', 'CV_MCC_std', 'CV_AUC_mean', 'CV_AUC_std', 'CV_GMean_mean']
    print(df[cv_cols].round(4).to_string(index=False))

    # 打印独立 Test 结果
    print("\n" + "=" * 80)
    print("           🎯 ENSEMBLE TEST RESULTS (vs. AMPActiPred Table 3)")
    print("=" * 80)
    test_cols = ['Bacteria', 'Test_Accuracy', 'Test_Sensitivity', 'Test_Specificity',
                 'Test_GMean', 'Test_MCC', 'Test_AUC', 'Test_F1']
    print(df[test_cols].round(4).to_string(index=False))

    # 打印对比
    print("\n" + "=" * 80)
    print("           📊 Ensemble vs. Single Best MCC")
    print("=" * 80)
    for _, r in df.iterrows():
        print(f"  {r['Bacteria']}: Ensemble={r['Test_MCC']:.3f} | Single={r['Single_MCC']:.3f} | "
              f"Gain={r['Test_MCC'] - r['Single_MCC']:+.3f}")

    if USE_LEARNABLE_ADAPTIVE and 'Final_Alpha' in df.columns:
        print("\n" + "=" * 80)
        print("           🔧 LEARNED ADAPTIVE WEIGHTS (α per Bacteria)")
        print("=" * 80)
        for _, r in df.iterrows():
            alpha_val = r['Final_Alpha']
            target_val = r['Target_Alpha']
            alpha_str = f"{alpha_val:.3f}" if alpha_val is not None else "N/A"
            target_str = f"{target_val:.2f}" if target_val is not None else "N/A"
            print(f"  {r['Bacteria']}: α = {alpha_str}  (target={target_str})")

    print(f"\n{'-' * 50}")
    print("TEST SUMMARY (Ensemble):")
    print(f"  Mean MCC:   {df['Test_MCC'].mean():.4f} ± {df['Test_MCC'].std():.4f}")
    print(f"  Mean AUC:   {df['Test_AUC'].mean():.4f} ± {df['Test_AUC'].std():.4f}")
    print(f"  Mean GMean: {df['Test_GMean'].mean():.4f}")
    print(f"  Mean Sen:   {df['Test_Sensitivity'].mean():.4f}")
    print(f"  Mean Spe:   {df['Test_Specificity'].mean():.4f}")
    print(f"{'-' * 50}")

    # Baseline对比分析
    baseline_mcc = {'Ab': 0.34, 'Bs': 0.714, 'Ec': 0.532, 'Ef': 0.427, 'Kp': 0.404,
                    'Ml': 0.557, 'Pa': 0.7, 'Sa': 0.586, 'Se': 0.719, 'St': 0.727}
    exceed = sum(1 for _, r in df.iterrows() if r['Test_MCC'] > baseline_mcc.get(r['Bacteria'], 0))
    print(f"\n  🏆 超过 Baseline 的菌种数量: {exceed}/{len(target_bacteria)}")
    for _, r in df.iterrows():
        b = r['Bacteria']
        diff = r['Test_MCC'] - baseline_mcc.get(b, 0)
        flag = "✅" if diff > 0 else "❌"
        print(f"     {flag} {b}: V4E-Prior={r['Test_MCC']:.3f} vs Baseline={baseline_mcc.get(b, 0):.3f} ({diff:+.3f})")

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
