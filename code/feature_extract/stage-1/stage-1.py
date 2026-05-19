import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ==================== 路径配置（改为相对路径） ====================
# 当前文件所在目录: code/feature_extract/stage-1/
CURRENT_DIR = Path(__file__).resolve().parent
# 项目根目录: AMPpred (graduation design)/
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent

# 模型路径
MODEL_PATH = PROJECT_ROOT / "Rostlab" / "esm2_t12_35M_UR50D"
# 数据路径
TRAIN_CSV = PROJECT_ROOT / "dataset" / "stage_1" / "Stage1_Train.csv"
# 输出路径（明确保存到当前目录下的 Train/ 文件夹）
OUTPUT_DIR = CURRENT_DIR / "Train"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PKL = OUTPUT_DIR / "stage1_features.pkl"

# ==================== 设备 & 模型加载 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Loading model from {MODEL_PATH}...")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), do_lower_case=False)
model = AutoModel.from_pretrained(str(MODEL_PATH)).to(device)
model.eval()


def extract_features(sequence, tokenizer, model, device):
    """
    取ESM-2的[CLS] token表示（480维）
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


# ==================== 加载数据 & 提取特征 ====================
print(f"Loading data from {TRAIN_CSV}...")
df = pd.read_csv(str(TRAIN_CSV))
print(f"Loaded {len(df)} sequences")

print("Extracting features...")
features = []
for i, seq in enumerate(df['Sequence']):
    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{len(df)}")
    feat = extract_features(seq, tokenizer, model, device)
    features.append(feat)

features = np.vstack(features)  # shape: (n_samples, 480)
labels = df['label'].values

# 保存到明确的相对路径
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump((features, labels), f)

print(f"\n✅ Stage-1特征提取完成！")
print(f"特征保存路径: {OUTPUT_PKL}")
print(f"特征形状: {features.shape}")
print(f"样本数量: {len(labels)}")
print(f"AMP比例: {labels.mean():.4f}")
