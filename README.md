
# AMPpred: 抗菌肽多病原体活性预测

基于多任务学习与自适应损失的两阶段深度学习模型。

---

## 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：`torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `matplotlib`

---


项目结构

```bash
AMPpred/
├── code/
│   ├── train/
│   │   ├── model.py              # CNN-BiLSTM-CBAM 网络 + 自适应损失定义
│   │   ├── stage-1/              # AMP/non-AMP 二分类训练与测试脚本
│   │   └── stage-2/
│   │       ├── V1/               # 单任务 + BCE
│   │       ├── V2/               # 单任务 + 全局 Focal
│   │       ├── V3/               # 单任务 + 全局 Focal + MIC 辅助
│   │       ├── V4/               # 单任务 + 自适应损失 + MIC 辅助
│   │       ├── V5/               # 单任务 + 自适应损失
│   │       ├── V6/               # 多任务共享 + 自适应损失（最优）
│   │       └── V7/               # 多任务共享 + 自适应损失 + MIC 辅助
│   ├── feature_extract/
│   │   ├── stage-1/              # Stage-1 ESM-2 特征提取脚本
│   │   └── stage-2/              # Stage-2 十菌种 ESM-2 特征提取脚本
├── dataset/                      # 数据集目录
│   ├── stage_1/
│   │   ├── Stage1_Train.csv
│   │   └── Stage1_Test.csv
│   └── stage_2/
│       ├── Train/                # 10 个菌种的训练数据
│       ├── Test/                 # 10 个菌种的测试数据
│       ├── addition/             # 补充数据或中间文件
│       └── 10个菌种正负样本统计结果.csv
├── picture/                      # 结果图片与可视化
├── Rostlab/                      # ESM-2 预训练权重（需自行下载）
├── .gitignore
├── requirements.txt
└── README.md
```

---


快速开始

1. 准备数据

你的 `dataset/` 目录已按以下结构组织：

```
dataset/
├── stage_1/
│   ├── Stage1_Train.csv
│   └── Stage1_Test.csv
└── stage_2/
    ├── Train/
    ├── Test/
    ├── addition/
    └── 10个菌种正负样本统计结果.csv
```

Stage-1 CSV 格式要求：包含肽序列与 AMP 标签（1=抗菌肽，0=非抗菌肽）。

Stage-2 CSV 格式要求：`Train/` 和 `Test/` 下按 10 个菌种分子目录（如 `Ab/`、`Sa/` 等），每个 CSV 包含序列、二分类活性标签、以及 `logMIC` 值（用于部分版本的辅助任务）。

2. 下载 ESM-2 预训练模型

```bash
mkdir -p Rostlab
git clone https://huggingface.co/facebook/esm2_t30_150M_UR50D Rostlab/esm2_t30_150M_UR50D
```

若未安装 `git-lfs`，请先安装。也可以手动从 Hugging Face 下载 `.bin` 权重文件放入 `Rostlab/`。

3. 离线特征提取

所有 ESM-2 特征统一离线提取为 `.npy`，避免训练时重复计算。

```bash
# Stage-1 特征提取
cd code/feature_extract/stage-1
python extract_esm2.py \
    --input ../../../dataset/stage_1/ \
    --output ../../../dataset/stage_1/features/ \
    --model_dir ../../../Rostlab/esm2_t30_150M_UR50D

# Stage-2 特征提取（10 个菌种）
cd code/feature_extract/stage-2
for b in Ab Bs Ec Ef Kp Ml Pa Sa Se St; do
    python extract_esm2.py \
        --input ../../../dataset/stage_2/Train/$b/ \
        --output ../../../dataset/stage_2/Train/$b/features/ \
        --model_dir ../../../Rostlab/esm2_t30_150M_UR50D
    python extract_esm2.py \
        --input ../../../dataset/stage_2/Test/$b/ \
        --output ../../../dataset/stage_2/Test/$b/features/ \
        --model_dir ../../../Rostlab/esm2_t30_150M_UR50D
done
```

提取后特征维度：`[B, 100, 480]`（B=批量，100=序列长度，480=ESM-2 隐藏维度）。

4. 训练

Stage-1（二分类筛选）：

```bash
cd code/train/stage-1
python train.py \
    --feature_dir ../../../dataset/stage_1/features/ \
    --epochs 150 \
    --lr 2e-3 \
    --batch_size 32 \
    --n_splits 5 \
    --seed 42
```

Stage-2（以最优版本 V6 为例）：

```bash
cd code/train/stage-2/V6
python train_mtl.py \
    --feature_root ../../../dataset/stage_2/ \
    --bacteria Ab Bs Ec Ef Kp Ml Pa Sa Se St \
    --pos_ratio_target 0.3 \
    --tasks_per_step 3 \
    --epochs 150 \
    --lr 2e-3 \
    --seed 42
```

---

关键参数说明

| 参数 | 含义 | 建议值  |
   |------|------|------|
| `pos_ratio_target` | 动态重采样后每批次正样本目标比例 | 0.3  |
| `tasks_per_step` | Round-Robin 每步随机抽取任务数 | 3    |
| `lr` | AdamW 初始学习率 | 2e-3 |
| `batch_size` | 训练批次大小 | 32   |
| `n_splits` | 交叉验证折数 | 5    |


---

注意事项

1. 路径问题：所有脚本使用相对路径，请在对应目录下运行（如 `code/train/stage-2/V6/`）。若需在其他位置运行，请修改脚本中的 `CODE_DIR` 指向 `code/train/`。
2. 大文件排除：`dataset/`、`Rostlab/`、以及提取后的 `*.npy` 已加入 `.gitignore`，不会上传 GitHub，需自行准备或本地保留。
3. 显存占用：Stage-2 多任务训练峰值显存约 8-12GB，若显存不足可减少 `tasks_per_step` 至 2。
4. 复现性：已固定 `random_state=42`（Sklearn）与 `torch.manual_seed(42+fold)`（PyTorch），配合离线静态 ESM-2 特征，确保结果可复现。
5. 模型定义位置：`code/train/model.py` 是 Stage-1 和 Stage-2 共用的网络定义文件，Stage-2 各版本（V1-V7）均通过相对路径引用该文件。

---

.gitignore 参考

```gitignore
# 数据集与特征（体积大，不上传）
dataset/
*.npy
*.npz
*.csv
*.fasta

# 预训练模型权重
Rostlab/
*.bin
*.pt
*.pth
*.safetensors
model/

# 训练输出
checkpoints/
runs/
logs/
__pycache__/
*.pyc
.ipynb_checkpoints/

# 系统文件
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
```

---

引用

若使用本项目，请引用相关基础工作：

- ESM-2: `Lin et al., Science, 2023`
- AMPActiPred 数据集: `Yao et al., Protein Science, 2024`

```