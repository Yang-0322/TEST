import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import os.path
import torch.nn.functional as F

from .occlusion_aware import OcclusionRobustFeatureExtractor, OcclusionConsistencyLoss, MultiScaleOcclusionAttention
from mamba.mamba_ssm.modules.bimamba import BiMamba
import clip.clip as clip


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path1 = '/home/xiaolei/projects/baseline/ReID/CLIMB-ReID/pretrain-models/clip/ViT-B-16.pt'
    model_path2 = '/YCY/Pretrained_models/ViT-B-16.pt'
    if os.path.exists(model_path1):
        model_path = model_path1
    elif os.path.exists(model_path2):
        model_path = model_path2
    else:
        model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


class CLIMB_OcclusionRobust(nn.Module):
    """遮挡鲁棒的 CLIMB 模型

    在原始 CLIMB 基础上添加：
    1. 遮挡感知注意力
    2. 遮挡一致性损失
    3. 自适应特征聚合
    """

    def __init__(self, num_classes, camera_num, view_num, cfg, use_occlusion_aware=True):
        super(CLIMB_OcclusionRobust, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.use_occlusion_aware = use_occlusion_aware

        self.in_planes = 768
        self.in_planes_proj = 512
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

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

        # CLIP 编码器
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        clip_model = load_clip_to_cpu(
            self.model_name,
            self.h_resolution,
            self.w_resolution,
            self.vision_stride_size
        )
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        # 冻结 patch projection 层
        for _, v in self.image_encoder.conv1.named_parameters():
            v.requires_grad_(False)
        print('Freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))

        # SIE (Shallow Instance Embedding)
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

        # ═══════════════════════════════════════════════════════
        # 遮挡感知模块 (新增)
        # ═══════════════════════════════════════════════════════
        if self.use_occlusion_aware:
            print("=== 启用遮挡感知机制 ===")

            # 遮挡鲁棒特征提取器 (在 Mamba 输出后)
            self.occlusion_robust_extractor = OcclusionRobustFeatureExtractor(
                dim=768,
                num_heads=8,
                mlp_ratio=4.,
                drop=0.
            )

            # 多尺度遮挡感知注意力
            self.multi_scale_occlusion_attn = MultiScaleOcclusionAttention(dim=768)

            # 遮挡一致性损失
            self.occlusion_consistency_loss = OcclusionConsistencyLoss(temperature=0.07)

            # 自适应聚合权重
            self.adaptive_fusion_weight = nn.Parameter(torch.tensor(0.5))

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
        """软排序"""
        B, N = scores.shape
        sorted_scores, _ = torch.sort(scores, descending=True)
        scores_exp = scores.unsqueeze(1)
        sorted_exp = sorted_scores.unsqueeze(2)
        dist = torch.abs(scores_exp - sorted_exp)
        P = torch.softmax(-dist / tau, dim=-1)
        return P

    def forward(self, x, get_image=False, cam_label=None, view_label=None):
        """前向传播"""
        # 只使用CLIP提取图像特征（用于构建记忆库）
        if get_image == True:
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

            _, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

            feat = self.bottleneck(img_feature)
            feat_proj = self.bottleneck_proj(img_feature_proj)
            out_feat = torch.cat([feat, feat_proj], dim=1)
            return out_feat

        # SIE 嵌入
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

        # CLIP 编码
        _, image_features, image_features_proj = self.image_encoder(x, cv_embed)
        img_feature = image_features[:, 0]
        img_feature_proj = image_features_proj[:, 0]

        # CLIP 分支
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        out_feat = torch.cat([feat, feat_proj], dim=1)

        # Mamba 分支
        feats_for_mamba = image_features.detach()
        feats_for_mamba_sp = feats_for_mamba[:, 1:, :]
        feats_for_mamba_cls = feats_for_mamba[:, 0, :]

        # 重排序
        re_order_mamba_sp = self.reorder_topology(feats_for_mamba_cls, feats_for_mamba_sp)

        # BiMamba 处理
        mamba_sp_out = self.sp_mamba_bi(re_order_mamba_sp)
        mamba_sp_out = torch.cat((feats_for_mamba_cls.unsqueeze(1), mamba_sp_out), dim=1)

        # ═══════════════════════════════════════════════════════
        # 遮挡感知处理 (新增)
        # ═══════════════════════════════════════════════════════
        occlusion_map = None
        if self.use_occlusion_aware:
            # 遮挡鲁棒特征提取
            mamba_sp_out, occlusion_score = self.occlusion_robust_extractor(self.norm2_mamba(mamba_sp_out))

            # 多尺度遮挡感知
            ms_out, occlusion_map = self.multi_scale_occlusion_attn(
                mamba_sp_out,
                h=self.h_resolution,
                w=self.w_resolution
            )

            # 自适应融合
            mamba_sp_out = self.adaptive_fusion_weight * mamba_sp_out + (1 - self.adaptive_fusion_weight) * ms_out
        else:
            # 原始处理
            mamba_sp_out2 = self.norm2_mamba(mamba_sp_out)

        # 空间注意力聚合
        A = self.sp_attention(mamba_sp_out)
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1)
        mamba_sp_out2 = torch.bmm(A, mamba_sp_out)
        mamba_sp_out2 = mamba_sp_out2.squeeze(1)

        feat_sp = self.bottleneck_proj_sp(mamba_sp_out2)

        # 输出
        if self.training:
            logit = self.classifier(out_feat)
            logitsp = self.classifier2(feat_sp)
            # 获取 use_occlusion_aware（兼容 DataParallel）
            use_occlusion_aware = getattr(self, 'use_occlusion_aware', False)
            if use_occlusion_aware and occlusion_map is not None:
                return out_feat, logit, feat_sp, logitsp, occlusion_map
            else:
                return out_feat, logit, feat_sp, logitsp
        else:
            feat_concat = torch.cat((out_feat, feat_sp), dim=1)
            use_occlusion_aware = getattr(self, 'use_occlusion_aware', False)
            if use_occlusion_aware and occlusion_map is not None:
                return feat_concat, out_feat, feat_sp, occlusion_map
            else:
                return feat_concat, out_feat, feat_sp

    def compute_occlusion_consistency_loss(self, feat1, feat2, occlusion_map1, occlusion_map2):
        """计算遮挡一致性损失"""
        if not self.use_occlusion_aware:
            return 0.0

        # 计算整体遮挡分数
        occlusion_score1 = occlusion_map1.mean(dim=[2, 3]) if occlusion_map1 is not None else torch.zeros(feat1.shape[0])
        occlusion_score2 = occlusion_map2.mean(dim=[2, 3]) if occlusion_map2 is not None else torch.zeros(feat2.shape[0])

        # 使用一致性损失
        loss = self.occlusion_consistency_loss(feat1, feat2, occlusion_score1, occlusion_score2)
        return loss

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


def make_model_occlusion_robust(cfg, num_classes, camera_num, view_num, use_occlusion_aware=True):
    """创建遮挡鲁棒的 CLIMB 模型"""
    model = CLIMB_OcclusionRobust(
        num_classes,
        camera_num,
        view_num,
        cfg,
        use_occlusion_aware=use_occlusion_aware
    )
    return model
