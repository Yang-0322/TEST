import random
import math
import torch
import numpy as np
from PIL import Image, ImageDraw


class RandomErasingEnhanced:
    """增强版随机遮挡数据增强，专门针对遮挡严重的ReID任务

    Args:
        probability: 执行遮挡的概率
        mode: 'pixel' (用均值填充) 或 'random' (用随机值填充)
        max_count: 最大遮挡块数 (默认3块，模拟多重遮挡)
        max_area: 单块最大遮挡面积比例
        min_count: 最小遮挡块数
        occlusion_shapes: 遮挡形状 ['rect', 'circle', 'ellipse', 'mixed']
        device: 执行设备
    """

    def __init__(
        self,
        probability=0.5,
        mode='pixel',
        max_count=3,
        min_count=1,
        max_area=0.3,
        occlusion_shapes=['rect', 'circle', 'ellipse', 'mixed'],
        device='cpu'
    ):
        self.probability = probability
        self.mode = mode
        self.max_count = max_count
        self.min_count = min_count
        self.max_area = max_area
        self.occlusion_shapes = occlusion_shapes
        self.device = device

    def __call__(self, img):
        """
        Args:
            img: Tensor [C, H, W]
        Returns:
            Tensor: 增强后的图像
        """
        if random.uniform(0, 1) >= self.probability:
            return img

        # 随机决定遮挡块数
        count = random.randint(self.min_count, self.max_count)

        for _ in range(count):
            img = self._apply_random_occlusion(img)

        return img

    def _apply_random_occlusion(self, img):
        """应用单次随机遮挡"""
        if random.uniform(0, 1) >= self.probability:
            return img

        # 随机选择遮挡形状
        shape = random.choice(self.occlusion_shapes)
        if shape == 'mixed':
            shape = random.choice(['rect', 'circle', 'ellipse'])

        # 创建遮挡掩码
        _, H, W = img.shape
        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)

        if shape == 'rect':
            mask = self._generate_rect_mask(H, W)
        elif shape == 'circle':
            mask = self._generate_circle_mask(H, W)
        elif shape == 'ellipse':
            mask = self._generate_ellipse_mask(H, W)

        # 应用遮挡
        if self.mode == 'pixel':
            # 使用均值填充
            mean_value = 0.5  # CLIP normalize后的均值
            for c in range(img.shape[0]):
                img[c][mask] = mean_value
        else:  # random
            # 使用随机值填充
            random_values = torch.rand(img.shape[0], mask.sum(), device=self.device)
            for c in range(img.shape[0]):
                img[c][mask] = random_values[c]

        return img

    def _generate_rect_mask(self, H, W):
        """生成矩形遮挡掩码"""
        area = H * W
        target_area = random.uniform(0.02, self.max_area) * area
        aspect_ratio = random.uniform(0.3, 3.3)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w > W:
            w = W
        if h > H:
            h = H

        x1 = random.randint(0, H - h)
        y1 = random.randint(0, W - w)

        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        mask[x1:x1 + h, y1:y1 + w] = True
        return mask

    def _generate_circle_mask(self, H, W):
        """生成圆形遮挡掩码"""
        area = H * W
        target_area = random.uniform(0.02, self.max_area) * area
        radius = int(round(math.sqrt(target_area / math.pi)))

        if radius * 2 > min(H, W):
            radius = min(H, W) // 2

        cx = random.randint(radius, W - radius)
        cy = random.randint(radius, H - radius)

        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)

        # 创建圆形掩码
        y, x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

        return mask

    def _generate_ellipse_mask(self, H, W):
        """生成椭圆遮挡掩码（模拟人体/篮球遮挡形状）"""
        area = H * W
        target_area = random.uniform(0.02, self.max_area) * area
        aspect_ratio = random.uniform(0.5, 2.0)

        # 计算椭圆的半轴
        a = int(round(math.sqrt(target_area * aspect_ratio / math.pi)))
        b = int(round(math.sqrt(target_area / (aspect_ratio * math.pi))))

        if a * 2 > W:
            a = W // 2
        if b * 2 > H:
            b = H // 2

        cx = random.randint(a, W - a)
        cy = random.randint(b, H - b)

        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)

        # 创建椭圆掩码
        y, x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        mask = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1

        return mask


class OcclusionAwareMixup:
    """遮挡感知的 Mixup 数据增强

    对遮挡严重的图像赋予更高的 mixup 权重
    """

    def __init__(self, alpha=0.2, beta=2.0, device='cpu'):
        self.alpha = alpha
        self.beta = beta
        self.device = device

    def __call__(self, img, label, img_mixed, label_mixed):
        """
        Args:
            img: Tensor [C, H, W]
            label: int
            img_mixed: Tensor [C, H, W]
            label_mixed: int
        Returns:
            mixed_img, mixed_label
        """
        if random.uniform(0, 1) > 0.5:
            return img, label

        # 从 Beta 分布采样混合比例
        lam = np.random.beta(self.alpha, self.beta)

        # 检测遮挡程度（简化版：检测大面积纯色区域）
        occlusion_score = self._estimate_occlusion(img)

        # 对遮挡严重的图像，调整混合权重
        if occlusion_score > 0.3:
            lam = lam * 0.7  # 降低遮挡图像的权重

        # 执行 mixup
        mixed_img = lam * img + (1 - lam) * img_mixed
        mixed_label = lam * label + (1 - lam) * label_mixed

        return mixed_img, mixed_label

    def _estimate_occlusion(self, img):
        """估计图像的遮挡程度"""
        # 简化方法：统计接近均值（0.5）的像素比例
        # 在实际应用中，可以用更复杂的遮挡检测算法
        mask = (img > 0.48) & (img < 0.52)
        occlusion_ratio = mask.float().mean().item()
        return occlusion_ratio


class PatchDropout:
    """Patch Dropout - 随机丢弃部分 patch token

    模拟部分 patch 信息丢失的情况，提高模型对遮挡的鲁棒性
    """

    def __init__(self, prob=0.1, max_blocks=2):
        """
        Args:
            prob: 每个 patch 被丢弃的概率
            max_blocks: 最大连续丢弃的 block 数
        """
        self.prob = prob
        self.max_blocks = max_blocks

    def __call__(self, patch_tokens, cls_token=None):
        """
        Args:
            patch_tokens: Tensor [B, N, C] - patch tokens
            cls_token: Tensor [B, 1, C] - CLS token (可选)
        Returns:
            Tensor: 处理后的 tokens
        """
        B, N, C = patch_tokens.shape
        device = patch_tokens.device

        # 生成二进制掩码
        keep_mask = torch.rand(N, device=device) > self.prob

        # 应用 blockwise dropout (模拟大面积遮挡)
        if self.max_blocks > 1:
            for _ in range(random.randint(0, self.max_blocks)):
                start = random.randint(0, N - 4)
                end = start + random.randint(2, 4)
                keep_mask[start:end] = False

        # 保留至少 50% 的 patch
        if keep_mask.sum() < N // 2:
            keep_mask[:N // 2] = True

        # 应用掩码
        kept_tokens = patch_tokens[:, keep_mask, :]

        if cls_token is not None:
            # 保留 CLS token
            return torch.cat([cls_token, kept_tokens], dim=1)
        else:
            return kept_tokens


class SpatialAttentionAugmentation:
    """空间注意力增强

    通过调整注意力权重，模拟遮挡对注意力的影响
    """

    def __init__(self, attention_dropout_prob=0.1):
        self.attention_dropout_prob = attention_dropout_prob

    def __call__(self, attention_map):
        """
        Args:
            attention_map: Tensor [B, N, N] 或 [B, H, W]
        Returns:
            Tensor: 增强后的 attention map
        """
        if random.uniform(0, 1) > self.attention_dropout_prob:
            return attention_map

        # 随机选择要抑制的 patch
        B, N = attention_map.shape[0], attention_map.shape[1]
        num_drops = random.randint(1, N // 4)
        drop_indices = random.sample(range(N), num_drops)

        # 抑制这些 patch 的注意力权重
        attention_map[:, drop_indices, :] *= 0.1
        attention_map[:, :, drop_indices] *= 0.1

        # 归一化
        attention_map = attention_map / attention_map.sum(dim=-1, keepdim=True)

        return attention_map
