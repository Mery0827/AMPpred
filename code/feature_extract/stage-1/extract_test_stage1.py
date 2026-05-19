import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# ==================== 路径配置（改为相对路径） ====================
# 当前文件所在目录: code/feature_extract/stage-1/
CURRENT_DIR = Path(__file__).resolve().parent
# 项目根目录: AMPpred (graduation design)/
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent

# 模型路径
MODEL_PATH = PROJECT_ROOT / "Rostlab" / "esm2_t12_35M_UR50D"
# 测试集路径
TEST_CSV = PROJECT_ROOT / "dataset" / "stage_1" / "Stage1_Test.csv"
# 输出目录：当前目录下的 Test/ 文件夹（与你截图结构一致）
OUTPUT_DIR = CURRENT_DIR / "Test"
OUTPUT_PKL = OUTPUT_DIR / "stage1_test_features.pkl"
OUTPUT_CSV = OUTPUT_DIR / "Stage1_Test.csv"  # 清洗后的CSV

OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), do_lower_case=False)
model = AutoModel.from_pretrained(str(MODEL_PATH)).to(device)
model.eval()


def extract_features(sequence, tokenizer, model, device):
    """
    取 ESM-2 的 [CLS] token 表示（480维）
    """
    formatted_seq = " ".join(sequence)
    inputs = tokenizer(
        formatted_seq,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
    return cls_embedding  # shape: (480,)


# ==================== 1. 加载并清洗 CSV ====================
print(f"\n[INFO] 加载测试集: {TEST_CSV}")
df = pd.read_csv(str(TEST_CSV))
print(f"原始列数: {len(df.columns)}, 样本数: {len(df)}")

# 只保留 Sequence 和 label
cols_to_keep = ['Sequence']
has_label = 'label' in df.columns
if has_label:
    cols_to_keep.append('label')
    print("[INFO] 检测到 label 列，已保留（用于独立测试评估）")
else:
    print("[WARN] 未检测到 label 列，仅保留 Sequence")

df_clean = df[cols_to_keep].copy()

# 简单清洗
df_clean = df_clean.dropna(subset=['Sequence'])
df_clean['Sequence'] = df_clean['Sequence'].astype(str)
print(f"[INFO] 清洗后样本数: {len(df_clean)}")

# 保存清洗后的 CSV
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 清洗后的 CSV 已保存: {OUTPUT_CSV}  （仅 {len(cols_to_keep)} 列）")

# ==================== 2. 提取 ESM-2 特征 ====================
print("\n[INFO] 开始提取 ESM-2 [CLS] 特征...")
features = []
for i, seq in enumerate(df_clean['Sequence']):
    if (i + 1) % 100 == 0 or i == len(df_clean) - 1:
        print(f"  Processed {i + 1}/{len(df_clean)}")
    feat = extract_features(seq, tokenizer, model, device)
    features.append(feat)

features = np.vstack(features)  # shape: (n_samples, 480)

# ==================== 3. 保存 pkl ====================
if has_label:
    labels = df_clean['label'].values
else:
    labels = None

with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump((features, labels), f)

print(f"\n✅ 测试集特征提取完成！")
print(f"特征 pkl 保存路径: {OUTPUT_PKL}")
print(f"特征形状: {features.shape}  (必须是 n×480)")
if has_label:
    print(f"样本数量: {len(labels)}, AMP比例: {labels.mean():.4f}")
else:
    print(f"样本数量: {len(features)}")
