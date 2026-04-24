"""
CLIMB 模型 with Path-based Feature Swapping
基于 CLIP 特征的跨人员路径交换，用于增强遮挡鲁棒性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import os.path
import numpy as np

from .model import CLIMB, load_clip_to_cpu, weights_init_kaiming, weights_init_classifier
from mamba.mamba_ssm.modules.bimamba import BiMamba


class PathSwappedEncoder(nn.Module):
    """基于路径交换的特征编码器

    核心思想：
    1. 在 CLIP 提取特征的不同阶段，进行跨人员的特征交换
    2. 交换策略：相似度驱动的特征替换
    3. 增强模型对遮挡和视角变化的鲁棒性
    """

    def __init__(self, clip_visual_encoder, in_planes=768, swap_ratio=0.3,
                 swap_layers=[6, 11], use_hard_negative=True):
        super(PathSwappedEncoder, self).__init__()
        self.clip_visual = clip_visual_encoder
        self.in_planes = in_planes
        self.swap_ratio = swap_ratio
        self.swap_layers = swap_layers
        self.use_hard_negative = use_hard_negative

        # 特征投影层（用于跨人员匹配）
        self.feature_proj = nn.Sequential(
            nn.Linear(in_planes, in_planes // 4),
            nn.ReLU(inplace=False),
            nn.Linear(in_planes // 4, in_planes)
        )
        self.feature_proj.apply(weights_init_kaiming)

        # 重要性预测网络（用于决定哪些patch需要交换）
        self.importance_net = nn.Sequential(
            nn.Linear(in_planes, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 1)
        )

        print(f"PathSwappedEncoder initialized with swap_ratio={swap_ratio}, swap_layers={swap_layers}")

    def _get_similar_targets(self, features, labels, num_swap=2):
        """获取相似的目标特征用于交换

        Args:
            features: [B, N, C] batch中的特征
            labels: [B] 人员ID标签
            num_swap: 每个样本交换的目标数量

        Returns:
            swap_features: [B, num_swap, C] 用于交换的特征
            swap_indices: [B, num_swap] 交换的索引
        """
        B, N, C = features.shape

        # 使用CLS token进行人员匹配
        cls_features = features[:, 0, :]  # [B, C]
        cls_features = F.normalize(cls_features, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(cls_features, cls_features.t())  # [B, B]

        # 创建mask：只选择不同ID的样本
        label_matrix = labels.unsqueeze(1).expand(B, B)
        same_id_mask = (label_matrix == label_matrix.t())
        sim_matrix = sim_matrix.masked_fill(same_id_mask, -float('inf'))

        # 获取最相似的K个不同ID样本
        if self.use_hard_negative:
            # 使用困难负样本（相似度较低但不是最低的）
            k = min(num_swap + 2, B - 1)
            _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
            swap_indices = topk_indices[:, -num_swap:]  # 取倒数num_swap个（相对较难但不是最难）
        else:
            # 使用正样本（相似度最高的）
            _, topk_indices = torch.topk(sim_matrix, k=num_swap, dim=1)
            swap_indices = topk_indices

        # 获取交换的CLS特征
        swap_features = cls_features[swap_indices]  # [B, num_swap, C]

        return swap_features, swap_indices

    def _compute_patch_importance(self, features, occlusion_mask=None):
        """计算每个patch的重要性分数

        Args:
            features: [B, N, C] patch特征
            occlusion_mask: [B, N] 可选的遮挡mask

        Returns:
            importance: [B, N] 重要性分数
        """
        # 使用特征投影后的相似度作为重要性
        projected = self.feature_proj(features)  # [B, N, C]
        importance = self.importance_net(projected).squeeze(-1)  # [B, N]

        # 如果有遮挡mask，降低遮挡区域的交换概率
        if occlusion_mask is not None:
            importance = importance * (1 - occlusion_mask)

        return F.softmax(importance, dim=1)

    def _swap_features(self, features, swap_features, swap_indices, importance_scores):
        """根据重要性分数交换特征

        Args:
            features: [B, N, C] 原始特征
            swap_features: [B, num_swap, C] 用于交换的CLS特征
            swap_indices: [B, num_swap] 交换的索引
            importance_scores: [B, N] 重要性分数

        Returns:
            swapped_features: [B, N, C] 交换后的特征
        """
        B, N, C = features.shape
        num_swap = swap_indices.shape[1]

        # 根据重要性选择要交换的patch
        num_swap_patches = int(N * self.swap_ratio)
        _, top_patch_indices = torch.topk(importance_scores, k=num_swap_patches, dim=1)

        # 随机选择使用哪个交换特征
        swap_choice = torch.randint(0, num_swap, (B, num_swap_patches), device=features.device)
        selected_swap_features = torch.stack(
            [swap_features[b, swap_choice[b]] for b in range(B)]
        )  # [B, num_swap_patches, C]

        # 执行交换：CLS token保持不变，只交换patch tokens
        cls_token = features[:, 0:1, :]  # [B, 1, C]
        patch_tokens = features[:, 1:, :].clone()  # [B, N-1, C] - 克隆以避免原地修改

        # 调整patch索引（去掉CLS）
        adjusted_patch_indices = (top_patch_indices - 1).clamp(min=0)

        # 执行交换 - 使用scatter避免原地操作
        for b in range(B):
            for i, patch_idx in enumerate(adjusted_patch_indices[b]):
                if patch_idx >= 0 and patch_idx < patch_tokens.shape[1]:
                    patch_tokens[b, patch_idx] = selected_swap_features[b, i]

        swapped_features = torch.cat([cls_token, patch_tokens], dim=1)
        return swapped_features

    def forward(self, x, cv_emb=None, labels=None, return_attention=False):
        """前向传播

        Args:
            x: [B, C, H, W] 输入图像
            cv_emb: [B, D] 或 None 相机/视角嵌入
            labels: [B] 人员ID标签（用于特征交换）
            return_attention: bool 是否返回注意力信息

        Returns:
            features: 各层的输出特征
            swap_info: dict 交换信息（用于分析）
        """
        # 获取初始卷积输出
        x = self.clip_visual.conv1(x)  # [B, width, h, w]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, h*w]
        x = x.permute(0, 2, 1)  # [B, h*w, width]

        # 添加CLS token和位置编码
        x = torch.cat([
            self.clip_visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        if cv_emb is not None:
            x[:, 0] = x[:, 0] + cv_emb

        x = x + self.clip_visual.positional_embedding.to(x.dtype)
        x = self.clip_visual.ln_pre(x)

        # Transformer处理，在指定层进行特征交换
        x = x.permute(1, 0, 2)  # [L, B, C]

        swap_info = {}
        x11_output = None
        x12_output = None

        # 遍历所有transformer层
        for i, block in enumerate(self.clip_visual.transformer.resblocks):
            x = block(x)

            # 在指定层进行特征交换
            if i in self.swap_layers and labels is not None and self.training:
                x_permuted = x.permute(1, 0, 2).detach()  # [B, L, C] - detach避免梯度问题

                # 获取相似的目标特征
                swap_features, swap_indices = self._get_similar_targets(
                    x_permuted, labels, num_swap=2
                )

                # 计算重要性
                importance = self._compute_patch_importance(x_permuted)

                # 执行特征交换
                x_swapped = self._swap_features(
                    x_permuted, swap_features, swap_indices, importance
                )

                # 混合原始和交换的特征（使用detach后的原始特征）
                x_swapped_with_grad = x_swapped.detach()
                x = x.permute(1, 0, 2)  # [B, L, C]

                # 使用stop_gradient策略：交换后的特征不参与梯度传播
                x = x * (1 - self.swap_ratio) + x_swapped_with_grad * self.swap_ratio
                x = x.permute(1, 0, 2)  # [L, B, C]

                # 记录交换信息
                swap_info[f'layer_{i}'] = {
                    'swap_indices': swap_indices,
                    'importance': importance
                }

            # 保存第11层的输出
            if i == 10:  # 第11层（索引从0开始）
                x11_output = x

        # 保存最后的输出
        x12_output = x

        # 转置并归一化
        x11 = x11_output.permute(1, 0, 2)
        x12 = x12_output.permute(1, 0, 2)

        x12 = self.clip_visual.ln_post(x12)

        if self.clip_visual.proj is not None:
            xproj = x12 @ self.clip_visual.proj

        if return_attention:
            return x11, x12, xproj, swap_info
        else:
            return x11, x12, xproj


class CLIMB_PathSwap(nn.Module):
    """带有路径交换机制的CLIMB模型

    在CLIP特征提取的不同阶段，通过跨人员特征交换来增强模型对遮挡和视角变化的鲁棒性
    """

    def __init__(self, num_classes, camera_num, view_num, cfg,
                 swap_ratio=0.3, swap_layers=[6, 11]):
        super(CLIMB_PathSwap, self).__init__()
        self.model_name = cfg.MODEL.NAME

        self.in_planes = 768
        self.in_planes_proj = 512
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        self.swap_ratio = swap_ratio
        self.swap_layers = swap_layers

        # Bottleneck 层
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        # 分类器
        self.classifier = nn.Linear(self.in_planes_proj + self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier2 = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.bottleneck_proj_sp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_sp.bias.requires_grad_(False)
        self.bottleneck_proj_sp.apply(weights_init_kaiming)

        # CLIP 编码器配置
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        # 加载CLIP模型
        clip_model = load_clip_to_cpu(
            self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size
        )
        clip_model.to("cuda")

        # 使用路径交换编码器替换原始的visual encoder
        self.image_encoder = PathSwappedEncoder(
            clip_model.visual,
            in_planes=self.in_planes,
            swap_ratio=swap_ratio,
            swap_layers=swap_layers,
            use_hard_negative=True
        )

        # 冻结patch projection层
        for _, v in self.image_encoder.clip_visual.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(
            self.image_encoder.clip_visual.conv1.weight.shape))

        # SIE
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
        else:
            self.cv_embed = None

        # Mamba 分支
        self.sp_mamba_bi = nn.Sequential(
            nn.LayerNorm(768),
            BiMamba(
                d_model=768,
                d_state=16,
                d_conv=4,
                expand=2,
            ),
        )

        self.norm2_mamba = nn.LayerNorm(768)
        self.norm3_mamba = nn.LayerNorm(768)

        # 空间注意力
        self.sp_attention = nn.Sequential(
            nn.Linear(768, 192),
            nn.Tanh(),
            nn.Linear(192, 1)
        )

        # 拓扑卷积
        self.topology_conv = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1, groups=768),
            nn.BatchNorm2d(768),
            nn.GELU()
        )
        self.topo_alpha = nn.Parameter(torch.ones(1) * 0.5)

    def reorder_topology(self, reference, raw, tau=0.1):
        """基于拓扑的重排序"""
        B, N, C = raw.shape
        H, W = self.h_resolution, self.w_resolution

        # 局部拓扑增强
        feat_2d = raw.transpose(1, 2).reshape(B, C, H, W)
        topo_feat = self.topology_conv(feat_2d)
        topo_feat = topo_feat.flatten(2).transpose(1, 2)

        # 融合原始特征与拓扑特征
        enhanced_raw = raw + self.topo_alpha * topo_feat

        # 计算感知拓扑的相似度评分
        ref = F.normalize(reference, dim=-1)
        raw_n = F.normalize(enhanced_raw, dim=-1)
        scores = torch.einsum('bc,bnc->bn', ref, raw_n)

        # 软排序
        if hasattr(self, 'softsort'):
            P = self.softsort(scores)
        else:
            # 简化版排序
            P = F.softmax(scores.unsqueeze(-1), dim=-1)

        reordered = torch.bmm(P, raw)
        return reordered

    def softsort(self, scores, tau=0.1):
        """
        scores: [B, N]  importance scores
        return: P [B, N, N] soft permutation matrix
        """
        B, N = scores.shape

        # sort scores to get target positions (no gradient needed here)
        sorted_scores, _ = torch.sort(scores, descending=True)

        # pairwise distance |α_j - α_(i)|
        scores_exp = scores.unsqueeze(1)  # [B, 1, N]
        sorted_exp = sorted_scores.unsqueeze(2)  # [B, N, 1]

        dist = torch.abs(scores_exp - sorted_exp)  # [B, N, N]

        # soft permutation matrix
        P = torch.softmax(-dist / tau, dim=-1)

        return P

    def reorder_soft(self, reference, raw, tau=0.1):
        """
        reference: [B, C]   CLS token
        raw:       [B, N, C] patch tokens
        """

        # normalize
        ref = F.normalize(reference, dim=-1)
        raw_n = F.normalize(raw, dim=-1)

        # importance score (same as original)
        scores = torch.einsum('bc,bnc->bn', ref, raw_n)  # [B, N]

        # soft permutation matrix
        P = self.softsort(scores)  # [B, N, N]

        # soft reordered tokens
        reordered = torch.bmm(P, raw)  # [B, N, C]

        return reordered

    def reorder(self, reference, raw):
        """基于相似度的patch重排序"""
        # attention_map = attention_map.mean(axis=1)  # torch.Size([64, 50, 50])
        reference_norm = F.normalize(reference, dim=-1).unsqueeze(1)  # bt, 1, 768
        raw_norm = F.normalize(raw, dim=-1)  # bt, 128, 768
        raw_norm = torch.transpose(raw_norm, 1, 2) # bt, 768, 128
        sim = torch.bmm(reference_norm, raw_norm).squeeze(1)  # [bt, 1, 768] [bt, 768, 128]= [bt, 1, 128]

        sorted, indices = torch.sort(sim, descending=True)

        selected_patch_embedding = []
        for i in range(indices.size(0)):   # bs
            all_patch_embeddings_i = raw[i, :, :].squeeze()  # torch.Size([128, 768])
            top_k_embedding = torch.index_select(all_patch_embeddings_i, 0, indices[i])  # torch.Size([128, 768])
            top_k_embedding = top_k_embedding.unsqueeze(0)  # torch.Size([1, 128, 768])
            selected_patch_embedding.append(top_k_embedding)
        selected_patch_embedding = torch.cat(selected_patch_embedding, 0)  # torch.Size([64, 128, 768])

        return selected_patch_embedding

    def forward(self, x, get_image=False, cam_label=None, view_label=None, labels=None):
        """前向传播"""
        # SIE嵌入
        if self.cv_embed is not None:
            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
        else:
            cv_embed = None

        # CLIP编码（使用路径交换机制）
        _, image_features, image_features_proj = self.image_encoder(x, cv_emb=cv_embed, labels=labels)
        img_feature = image_features[:, 0]
        img_feature_proj = image_features_proj[:, 0]

        # CLIP分支
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        out_feat = torch.cat([feat, feat_proj], dim=1)

        # Mamba分支
        feats_for_mamba = image_features.detach()
        feats_for_mamba_sp = feats_for_mamba[:, 1:, :]
        feats_for_mamba_cls = feats_for_mamba[:, 0, :]

        # 重排序
        # re_order_mamba_sp = self.reorder(feats_for_mamba_cls, feats_for_mamba_sp)
        re_order_mamba_sp = self.reorder_topology(feats_for_mamba_cls, feats_for_mamba_sp)

        # BiMamba处理
        mamba_sp_out = self.sp_mamba_bi(re_order_mamba_sp)
        mamba_sp_out = torch.cat((feats_for_mamba_cls.unsqueeze(1), mamba_sp_out), dim=1)

        # 空间注意力聚合
        mamba_sp_out2 = self.norm2_mamba(mamba_sp_out)
        A = self.sp_attention(mamba_sp_out2)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)
        mamba_sp_out2 = torch.bmm(A, mamba_sp_out2)
        mamba_sp_out2 = mamba_sp_out2.squeeze(1)

        feat_sp = self.bottleneck_proj_sp(mamba_sp_out2)

        if get_image:
            return out_feat

        if self.training:
            logit = self.classifier(out_feat)
            logitsp = self.classifier2(feat_sp)
            return out_feat, logit, feat_sp, logitsp
        else:
            feat_concat = torch.cat((out_feat, feat_sp), dim=1)
            return feat_concat, out_feat, feat_sp

    def load_param(self, trained_path):
        """加载预训练参数"""
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if not self.training and 'classifier' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        """加载参数用于微调"""
        param_dict = torch.load(model_path)
        for i in param_dict:
            if i in self.state_dict():
                self.state_dict()[i].copy_(param_dict[i])
        else:
            print(f'Skipping {i} - not in model')
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model_patch_swap(cfg, num_classes, camera_num, view_num,
                         swap_ratio=0.3, swap_layers=[6, 11]):
    """创建带有路径交换机制的CLIMB模型"""
    model = CLIMB_PathSwap(
        num_classes,
        camera_num,
        view_num,
        cfg,
        swap_ratio=swap_ratio,
        swap_layers=swap_layers
    )
    return model
