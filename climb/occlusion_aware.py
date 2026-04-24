import torch
import torch.nn as nn
import torch.nn.functional as F


class OcclusionAwareAttention(nn.Module):
    """遮挡感知注意力模块

    通过学习每个 patch 的遮挡程度，自适应调整注意力权重
    降低对遮挡 patch 的依赖
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 遮挡预测头
        self.occlusion_predictor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        # 自适应权重
        self.adaptive_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Args:
            x: [B, N, C] - input tokens
        Returns:
            out: [B, N, C] - output tokens
            occlusion_score: [B, N, 1] - 遮挡分数
        """
        B, N, C = x.shape

        # 计算注意力
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # 预测遮挡分数 (基于特征的不确定性)
        occlusion_score = self._predict_occlusion(x)

        # 自适应调整输出：对遮挡严重的 patch 降低其影响力
        adaptive_out = out * (1 - occlusion_score * self.adaptive_weight)

        return adaptive_out, occlusion_score

    def _predict_occlusion(self, x):
        """预测每个 patch 的遮挡程度"""
        # 简化方法：基于特征的方差预测遮挡
        # 在实际应用中，可以使用更复杂的方法
        feat_var = x.var(dim=-1, keepdim=True)
        occlusion_score = self.occlusion_predictor(x)

        # 结合特征方差（遮挡区域通常方差较小）
        combined_score = occlusion_score * 0.7 + torch.sigmoid(-feat_var * 10) * 0.3

        return combined_score


class OcclusionRobustFeatureExtractor(nn.Module):
    """遮挡鲁棒特征提取器

    结合多种机制提高对遮挡的鲁棒性
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0.):
        super().__init__()

        # 遮挡感知注意力
        self.occlusion_attn = OcclusionAwareAttention(
            dim, num_heads=num_heads,
            attn_drop=drop, proj_drop=drop
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, C] - input tokens
        Returns:
            out: [B, N, C] - output tokens
            occlusion_score: [B, N, 1] - 遮挡分数
        """
        # 遮挡感知注意力
        attn_out, occlusion_score = self.occlusion_attn(self.norm1(x))

        # 残差连接
        x = x + attn_out

        # MLP
        mlp_out = self.mlp(self.norm2(x))

        # 门控融合：根据遮挡程度调整 MLP 输出的贡献
        gate_weight = self.gate(torch.cat([x, mlp_out], dim=-1))
        out = x + mlp_out * gate_weight

        return out, occlusion_score


class OcclusionConsistencyLoss(nn.Module):
    """遮挡一致性损失

    确保模型在不同遮挡情况下学到一致的特征表示
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat1, feat2, occlusion_mask1, occlusion_mask2):
        """
        Args:
            feat1: [B, D] - 特征1
            feat2: [B, D] - 特征2 (不同遮挡版本)
            occlusion_mask1: [B] - 图像1的遮挡程度 (0-1)
            occlusion_mask2: [B] - 图像2的遮挡程度 (0-1)
        Returns:
            loss: 遮挡一致性损失
        """
        # L2 归一化
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        # 计算相似度
        similarity = (feat1 * feat2).sum(dim=-1)

        # 对遮挡严重的样本给予更高的权重
        occlusion_weight = (occlusion_mask1 + occlusion_mask2) / 2

        # 一致性损失：最大化相似度
        loss = (1 - similarity) * (1 + occlusion_weight)

        return loss.mean()


class MultiScaleOcclusionAttention(nn.Module):
    """多尺度遮挡感知注意力

    在不同尺度上检测和处理遮挡
    """

    def __init__(self, dim):
        super().__init__()

        # 多尺度特征提取
        self.scales = [1, 2, 4]
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=s, padding=s)
            for s in self.scales
        ])

        # 遮挡检测
        self.occlusion_detector = nn.Sequential(
            nn.Conv2d(dim * len(self.scales), dim // 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 自适应聚合
        self.aggregation = nn.Sequential(
            nn.Linear(dim * len(self.scales), dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, h=16, w=8):
        """
        Args:
            x: [B, N, C] - patch tokens
            h, w: 空间分辨率
        Returns:
            out: [B, N, C] - 输出 tokens（保持与输入相同的序列长度）
            occlusion_map: [B, 1, H, W] - 遮挡图
        """
        B, N, C = x.shape

        # 保存原始输入的完整形状（用于后续恢复）
        original_x = x
        has_cls_token = (C == 768 and N > h * w)

        # 计算实际的空间分辨率（排除 CLS token）
        N_without_cls = N - 1 if has_cls_token else N
        if N_without_cls != h * w:
            # 自动推断正确的 h 和 w
            import math
            h = int(math.sqrt(N_without_cls))
            w = N_without_cls // h

        # 提取 patch tokens（如果有 CLS token 则排除）
        if has_cls_token:
            cls_token = x[:, :1, :]  # 保存 CLS token
            x = x[:, 1:, :]  # 移除 CLS token
        else:
            cls_token = None

        # 重塑为 2D
        feat_2d = x.transpose(1, 2).reshape(B, C, h, w)

        # 多尺度特征
        multi_scale_feats = []
        for conv in self.scale_convs:
            scale_feat = conv(feat_2d)
            scale_feat = F.interpolate(scale_feat, size=(h, w), mode='bilinear', align_corners=False)
            multi_scale_feats.append(scale_feat)

        # 拼接多尺度特征
        concat_feat = torch.cat(multi_scale_feats, dim=1)

        # 检测遮挡
        occlusion_map = self.occlusion_detector(concat_feat)

        # 聚合多尺度特征
        flatten_feat = torch.cat([f.flatten(2) for f in multi_scale_feats], dim=1)
        flatten_feat = flatten_feat.transpose(1, 2)
        out = self.aggregation(flatten_feat)

        # 如果输入有 CLS token，则在输出中也添加 CLS token（使用原始 CLS token）
        if cls_token is not None:
            out = torch.cat([cls_token, out], dim=1)

        return out, occlusion_map
