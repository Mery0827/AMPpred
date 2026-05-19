import numpy as np
import pandas as pd
import torch
import pickle
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             matthews_corrcoef)

# ==================== 路径配置 ====================
# 当前文件所在目录: code/train/stage-1/
CURRENT_DIR = Path(__file__).resolve().parent
# model.py 所在目录: code/train/
CODE_DIR = CURRENT_DIR.parent
# 特征目录: code/feature_extract/stage-1/
FEATURE_DIR = CODE_DIR / "feature_extract" / "stage-1"

# 数据路径
TRAIN_PKL = FEATURE_DIR / "Train" / "stage1_features.pkl"  # 用于fit scaler
TEST_PKL = FEATURE_DIR / "Test" / "stage1_test_features.pkl"  # 独立测试集

# 模型路径前缀（stage1_fold1.pth ~ stage1_fold5.pth）
MODEL_PREFIX = "stage1_fold"
NUM_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入 model.py（假设在 code/ 目录下）
sys.path.insert(0, str(CODE_DIR))
from model import DeepAMPpred

# 输出结果路径
OUTPUT_CSV = CURRENT_DIR / "stage1_independent_test_results.csv"

# ==================== 1. 加载数据 ====================
print(f"[INFO] 加载训练集（用于标准化）: {TRAIN_PKL}")
if not TRAIN_PKL.exists():
    raise FileNotFoundError(f"找不到训练特征: {TRAIN_PKL}")

with open(TRAIN_PKL, 'rb') as f:
    X_train, y_train = pickle.load(f)
print(f"      训练集: {X_train.shape}, AMP比例={y_train.mean():.4f}")

print(f"[INFO] 加载独立测试集: {TEST_PKL}")
if not TEST_PKL.exists():
    raise FileNotFoundError(f"找不到测试特征: {TEST_PKL}")

with open(TEST_PKL, 'rb') as f:
    X_test, y_test = pickle.load(f)
print(f"      测试集: {X_test.shape}")

if y_test is None:
    raise ValueError("测试集缺少 label，无法评估。请重新提取特征时保留 label 列。")

# ==================== 2. 维度校验 ====================
if X_test.shape[1] != X_train.shape[1]:
    raise ValueError(
        f"维度不匹配！测试集{X_test.shape[1]}维 vs 训练集{X_train.shape[1]}维。 "
        f"请检查是否使用同一模型提取特征。"
    )

# ==================== 3. 标准化（仅用训练集统计量） ====================
scaler = StandardScaler()
scaler.fit(X_train)  # 只在训练集上 fit
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)

# ==================== 4. 5-Fold 模型集成预测 ====================
all_fold_probs = []

for fold in range(1, NUM_FOLDS + 1):
    model_path = CURRENT_DIR / f"{MODEL_PREFIX}{fold}.pth"
    print(f"[INFO] 加载模型: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件: {model_path}\n请先运行 stage-1(1).py 完成训练")

    model = DeepAMPpred(
        input_dim=X_train.shape[1],  # 480
        num_classes=2,
        multilabel=False,
        use_cnn2=True,
        lstm_layers=2,
        lstm_hidden=128,
        dropout=0.5
    ).to(DEVICE)

    model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_fold_probs.append(probs)

# ==================== 5. 集成：概率平均 ====================
avg_probs = np.mean(all_fold_probs, axis=0)
preds = (avg_probs >= 0.5).astype(int)

# ==================== 6. 计算评估指标 ====================
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

metrics = {
    'Accuracy': accuracy_score(y_test, preds),
    'Precision': precision_score(y_test, preds, zero_division=0),
    'Recall/Sensitivity': recall_score(y_test, preds, zero_division=0),
    'F1': f1_score(y_test, preds, zero_division=0),
    'MCC': matthews_corrcoef(y_test, preds),
    'AUC': roc_auc_score(y_test, avg_probs),
    'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
}

print("\n" + "=" * 55)
print("Stage-1 独立测试结果 (5-Fold 概率平均集成)")
print("=" * 55)
for k, v in metrics.items():
    print(f"  {k:20s}: {v:.4f}")

# ==================== 7. 保存结果 ====================
result_df = pd.DataFrame({
    'True': y_test,
    'Pred': preds,
    'Prob_AMP': avg_probs
})
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n[INFO] 预测结果已保存: {OUTPUT_CSV}")
