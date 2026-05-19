import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== CBAM 注意力模块 ======================
class ChannelAttentionModule(nn.Module):
    """CBAM通道注意力模块（1D适配版）"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        reduced_dim = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1))
        max_out = self.mlp(self.max_pool(x).squeeze(-1))
        out = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * out


class SpatialAttentionModule(nn.Module):
    """CBAM空间注意力模块（1D适配版）"""
    def __init__(self, kernel_size=7):
        super().__init__()
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module（1D版）"""
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ====================== 第一阶段模型（二分类 AMP/non-AMP）=====================
class DeepAMPpred(nn.Module):
    """
    第一阶段：区分 AMP 和 non-AMP
    输入: ESM-2 特征 [batch, seq_len=100, 480]
    """
    def __init__(self, input_dim=480, seq_len=100, num_classes=2, 
                 lstm_hidden=64, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.cbam = CBAM(in_channels=2 * lstm_hidden, reduction_ratio=16)
        
        classifier_input_dim = (2 * lstm_hidden) * seq_len
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x.permute(0, 2, 1)
        x = self.cbam(x)
        x = self.classifier(x)
        return x


# ====================== 第二阶段：10细菌抗菌活性 + MIC辅助任务 ======================
class DeepAMPpredStage2_MICAux(nn.Module):
    """
    第二阶段：10种细菌抗菌活性预测 + MIC值辅助任务
    """
    def __init__(self, input_dim=480, seq_len=100, num_bacteria=10,
                 lstm_hidden=64, lstm_layers=2, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_bacteria = num_bacteria
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.cbam = CBAM(in_channels=2 * lstm_hidden, reduction_ratio=16)
        
        classifier_input_dim = (2 * lstm_hidden) * seq_len
        
        self.shared_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.bacteria_heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_bacteria)
        ])
        
        # MIC 辅助任务分支：从 CBAM 输出提取特征
        self.mic_feature = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(2 * lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.mic_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, return_features=False, bacteria_idx=None):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x.permute(0, 2, 1)
        x = self.cbam(x)
        
        feat_mic = self.mic_feature(x)
        shared = self.shared_classifier(x)
        
        if bacteria_idx is not None:
            out_main = self.bacteria_heads[bacteria_idx](shared)
        else:
            outputs = [head(shared) for head in self.bacteria_heads]
            out_main = torch.cat(outputs, dim=1)
        
        if return_features:
            return out_main, feat_mic
        
        return out_main


# ====================== 标准 Focal Loss ======================
class FocalLoss(nn.Module):
    """Focal Loss for 类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


# ====================== V6-v6 / V7: Prior-Anchored Learnable α ======================
class AdaptiveBCEFocalLoss(nn.Module):
    """
    先验锚定的可学习 BCE-Focal 组合损失。
    
    Loss = α * BCE + (1 - α) * FocalLoss(focal_alpha=0.25)
    
    其中 α 通过 sigmoid(logit_alpha) 约束在 (0.05, 0.95)。
    在 logit 空间施加高斯先验正则化，防止共享 MTL 下 α 被梯度耦合同质化。
    """
    def __init__(self, gamma=1.5, focal_alpha=0.25, reg_weight=0.5, 
                 smoothing=0.1, init_logit=0.0, target_alpha=0.5):
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.tensor(float(init_logit)))
        self.gamma = gamma
        self.focal_alpha = focal_alpha
        self.reg_weight = reg_weight
        self.smoothing = smoothing
        
        # 先验 target 的 logit 值（预计算，不参与梯度）
        self.register_buffer('target_logit', torch.logit(torch.tensor(target_alpha)))
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    @property
    def alpha(self):
        """返回当前 α 值，范围 (0.05, 0.95)"""
        return 0.05 + 0.9 * torch.sigmoid(self.logit_alpha)

    def forward(self, inputs, targets):
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(-1)
        if targets.dim() > inputs.dim():
            targets = targets.squeeze(-1)

        with torch.no_grad():
            bce_hard = self.bce(inputs, targets)
            p_t = torch.exp(-bce_hard).clamp(min=1e-7, max=1.0 - 1e-7)
        
        if self.smoothing > 0:
            targets_smooth = targets * (1.0 - self.smoothing) + self.smoothing * 0.5
            bce_loss = self.bce(inputs, targets_smooth)
        else:
            bce_loss = self.bce(inputs, targets)

        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        a = self.alpha
        combined = a * bce_loss + (1 - a) * focal_loss
        
        # 关键：logit 空间先验正则化，将 α 钉在 target 附近
        alpha_reg = self.reg_weight * torch.pow(self.logit_alpha - self.target_logit, 2)

        return combined.mean() + alpha_reg
