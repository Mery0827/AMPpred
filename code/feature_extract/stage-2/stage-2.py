import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ===================== 路径配置（相对路径） =====================
# 当前文件所在目录: code/feature_extract/stage-2/
CURRENT_DIR = Path(__file__).resolve().parent
# 项目根目录: AMPpred (graduation design)/
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent

# 数据源
DATA_DIR = PROJECT_ROOT / "dataset" / "stage_2"

# 输出目录：当前脚本同级目录下的 Train/ 和 Test/
SAVE_DIR = CURRENT_DIR
TRAIN_SAVE = SAVE_DIR / "Train"
TEST_SAVE = SAVE_DIR / "Test"
TRAIN_SAVE.mkdir(exist_ok=True)
TEST_SAVE.mkdir(exist_ok=True)

BACTERIA_NAMES = ['Ab', 'Bs', 'Ec', 'Ef', 'Kp', 'Ml', 'Pa', 'Sa', 'Se', 'St']
MAX_LEN = 100
INPUT_DIM = 480
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 加载本地 ESM-2 =====================
MODEL_PATH = PROJECT_ROOT / "Rostlab" / "esm2_t12_35M_UR50D"
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), do_lower_case=False, local_files_only=True)
model = AutoModel.from_pretrained(str(MODEL_PATH), local_files_only=True)
model = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")


def extract_features(sequence):
    """提取整条序列特征 [MAX_LEN, 480]"""
    formatted_seq = " ".join(sequence)
    inputs = tokenizer(
        formatted_seq,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    seq_feature = outputs.last_hidden_state[0].cpu().numpy()
    return seq_feature


# ===================== 主函数 =====================
def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data source: {DATA_DIR}")
    print(f"Output dir: {SAVE_DIR}")
    print("-" * 50)

    for b in BACTERIA_NAMES:
        for split in ['Train', 'Test']:
            csv_path = DATA_DIR / split / f"{b}.csv"
            if not csv_path.exists():
                print(f"\n⚠️ 跳过: {csv_path} 不存在")
                continue

            print(f"\n{'='*50}")
            print(f"Processing: {b} / {split}")

            df = pd.read_csv(str(csv_path))

            # 自动探测列名
            seq_col = next((c for c in ['Sequence', 'sequence', 'SEQ', 'seq'] if c in df.columns), None)
            label_col = next((c for c in ['label', 'Label', 'LABEL'] if c in df.columns), None)
            mic_col = next((c for c in ['value', 'Value', 'mic', 'MIC', f'{b}_value'] if c in df.columns), None)

            if seq_col is None or label_col is None:
                raise ValueError(f"{b}.csv 缺少必要列！可用列: {list(df.columns)}")

            sequences = df[seq_col].astype(str).tolist()
            labels = df[label_col].values.astype(np.float32)

            if mic_col:
                mic = df[mic_col].fillna(0).values.astype(np.float32)
                print(f"  Found MIC: '{mic_col}'")
            else:
                mic = np.zeros(len(df), dtype=np.float32)
                print(f"  ⚠️ No MIC column")

            n_pos = int(labels.sum())
            print(f"  Samples: {len(sequences)} | Pos={n_pos} | Neg={len(labels)-n_pos}")

            # 提取特征
            X = []
            for seq in tqdm(sequences, desc=f"{b}-{split}", leave=False):
                X.append(extract_features(seq))

            X = np.array(X)
            print(f"  Feature shape: {X.shape}")

            # 保存到对应目录
            out_folder = TRAIN_SAVE if split == 'Train' else TEST_SAVE
            np.save(out_folder / f"{b}_features.npy", X)
            np.save(out_folder / f"{b}_labels.npy", labels)
            np.save(out_folder / f"{b}_mic.npy", mic)
            print(f"  ✅ Saved to {out_folder}/{b}_*.npy")

    print(f"\n{'='*50}")
    print("🎉 All features extracted!")
    print(f"Train: {TRAIN_SAVE}")
    print(f"Test:  {TEST_SAVE}")


if __name__ == '__main__':
    main()
