import numpy as np
import random
import torch
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef)
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ==================== 路径配置 ====================
# 当前文件所在目录: code/train/stage-1/
CURRENT_DIR = Path(__file__).resolve().parent
# model.py 所在目录: code/train/
CODE_DIR = CURRENT_DIR.parent
# 特征目录: code/feature_extract/stage-1/
FEATURE_DIR = CODE_DIR / "feature_extract" / "stage-1"

# 数据路径（明确指向 stage-1/Train/）
TRAIN_PKL = FEATURE_DIR / "Train" / "stage1_features.pkl"

# 模型和输出保存到当前脚本同级目录（train/stage-1/）
MODEL_SAVE_DIR = CURRENT_DIR
METRICS_CSV = CURRENT_DIR / "stage1_metrics.csv"
PREDICTIONS_CSV = CURRENT_DIR / "stage1_predictions.csv"

# 导入 model.py（假设在 code/ 目录下，与 feature_extract/ 同级）
sys.path.insert(0, str(CODE_DIR))
from model import DeepAMPpred, WeightedCrossEntropyLoss

random.seed(1)
np.random.seed(17)
torch.manual_seed(153)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['AUC'] = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return metrics, all_probs, all_preds, all_labels


def train_stage1():
    print(f"[INFO] 加载训练特征: {TRAIN_PKL}")
    if not TRAIN_PKL.exists():
        raise FileNotFoundError(
            f"找不到训练特征文件: {TRAIN_PKL}\n请先运行 feature_extract/stage-1/stage-1.py 提取特征")

    with open(TRAIN_PKL, 'rb') as f:
        X, y = pickle.load(f)

    print(f"Data loaded: X.shape={X.shape}, AMP ratio: {y.mean():.4f}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    input_dim = X.shape[1]
    print(f"Input dimension: {input_dim}")

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_results = []
    all_fold_probs, all_fold_labels, all_fold_preds = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f'\n{"=" * 50}')
        print(f'Fold {fold + 1}/5')
        print(f'{"=" * 50}')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

        model = DeepAMPpred(
            input_dim=input_dim,
            num_classes=2,
            multilabel=False,
            use_cnn2=True,
            lstm_layers=2,
            lstm_hidden=128,
            dropout=0.5
        ).to(device)

        # 类别权重
        pos_ratio = y_train.float().mean().item()
        neg_weight = 1.0 / (1 - pos_ratio)
        pos_weight = 1.0 / pos_ratio
        total = neg_weight + pos_weight
        class_weights = torch.FloatTensor([neg_weight / total * 2, pos_weight / total * 2])
        print(f"Class weights: {class_weights.numpy()}")

        criterion = WeightedCrossEntropyLoss(class_weights=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )

        num_epochs = 20
        best_auc = 0.0
        best_metrics = None

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            metrics, probs, preds, labels = evaluate(model, test_loader, device)
            scheduler.step(train_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch {epoch + 1:2d}: Loss={train_loss:.4f}, '
                      f'AUC={metrics["AUC"]:.4f}, F1={metrics["F1"]:.4f}, '
                      f'Sen={metrics["Sensitivity"]:.3f}, Spe={metrics["Specificity"]:.3f}')

            if metrics['AUC'] > best_auc:
                best_auc = metrics['AUC']
                best_metrics = metrics.copy()
                model_save_path = MODEL_SAVE_DIR / f'stage1_fold{fold + 1}.pth'
                torch.save(model.state_dict(), model_save_path)
                print(f'  💾 最佳模型已保存: {model_save_path}')

        fold_results.append(best_metrics)
        all_fold_probs.extend(probs)
        all_fold_preds.extend(preds)
        all_fold_labels.extend(labels)

        print(f'Fold {fold + 1} Best: AUC={best_metrics["AUC"]:.4f}')

    # 汇总
    print(f'\n{"=" * 50}')
    print('5-Fold Summary')
    print(f'{"=" * 50}')

    df = pd.DataFrame(fold_results)
    mean = df.mean()
    std = df.std()

    for metric in mean.index:
        print(f'  {metric:12s}: {mean[metric]:.4f} ± {std[metric]:.4f}')

    df.to_csv(METRICS_CSV, index=False)
    print(f"\n[INFO] 指标已保存: {METRICS_CSV}")

    pd.DataFrame({
        'True': all_fold_labels,
        'Pred': all_fold_preds,
        'Prob': all_fold_probs
    }).to_csv(PREDICTIONS_CSV, index=False)
    print(f"[INFO] 预测结果已保存: {PREDICTIONS_CSV}")


if __name__ == '__main__':
    train_stage1()
