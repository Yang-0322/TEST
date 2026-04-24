import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .dataset import ImageDataset, IterLoader
from .occlusion_augmentation import RandomErasingEnhanced
from .sampler import RandomIdentitySampler, RandomMultipleGallerySampler
from datasets.market1501 import Market1501
from datasets.qiuxiu import Qiuxiu
from datasets.msmt17 import MSMT17
import torch.distributed as dist
from datasets.sampler_ddp import RandomIdentitySampler_DDP


FACTORY = {
    'market1501': Market1501,
    'qiuxiu': Qiuxiu,
    'msmt17': MSMT17,
}


def train_collate_fn(batch):
    """
    训练时的 collate 函数
    """
    if len(batch[0]) == 5:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
    elif len(batch[0]) == 4:
        # 处理只有 4 个值的情况（可能是默认 collate_fn 的问题）
        imgs, pids, camids, viewids = zip(*batch)
        img_paths = None
    else:
        raise ValueError(f"Unexpected number of values in batch: {len(batch[0])}")
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    """
    验证时的 collate 函数
    """
    if len(batch[0]) == 5:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
    elif len(batch[0]) == 4:
        # 处理只有 4 个值的情况
        imgs, pids, camids, viewids = zip(*batch)
        img_paths = None
    else:
        raise ValueError(f"Unexpected number of values in batch: {len(batch[0])}")
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    camids_batch = camids  # 与 camids 相同，保持与原始代码一致
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader_with_occlusion(cfg):
    """
    创建带有遮挡感知数据增强的 dataloader

    Args:
        cfg: 配置对象，需要包含以下字段:
            - INPUT.SIZE_TRAIN: 训练图像尺寸
            - INPUT.PROB: 水平翻转概率
            - INPUT.PADDING: 填充大小
            - INPUT.PIXEL_MEAN: 像素均值
            - INPUT.PIXEL_STD: 像素标准差
            - INPUT.RE_PROB: 随机遮挡概率
            - INPUT.OCCLUSION_MODE: 遮挡模式 'enhanced' 或 'original'
            - INPUT.OCCLUSION_MAX_COUNT: 最大遮挡块数
            - INPUT.OCCLUSION_SHAPES: 遮挡形状列表
            - DATASETS.NAMES: 数据集名称
            - DATASETS.ROOT_DIR: 数据集根目录
            - DATALOADER.NUM_WORKERS: worker 数量
            - DATALOADER.NUM_INSTANCE: 每个 ID 的实例数
            - SOLVER.IMS_PER_BATCH: batch size
    """
    # 基础变换
    base_transforms = [
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ]

    # 添加遮挡增强
    if hasattr(cfg.INPUT, 'OCCLUSION_MODE') and cfg.INPUT.OCCLUSION_MODE == 'enhanced':
        # 使用增强版遮挡
        max_count = getattr(cfg.INPUT, 'OCCLUSION_MAX_COUNT', 3)
        occlusion_shapes = getattr(cfg.INPUT, 'OCCLUSION_SHAPES',
                                  ['rect', 'circle', 'ellipse', 'mixed'])

        occlusion_aug = RandomErasingEnhanced(
            probability=cfg.INPUT.RE_PROB,
            mode='pixel',
            max_count=max_count,
            min_count=1,
            max_area=0.3,
            occlusion_shapes=occlusion_shapes,
            device='cpu'
        )
        base_transforms.append(occlusion_aug)
    else:
        # 使用原始遮挡
        from timm.data.random_erasing import RandomErasing
        occlusion_aug = RandomErasing(
            probability=cfg.INPUT.RE_PROB,
            mode='pixel',
            max_count=1,
            device='cpu'
        )
        base_transforms.append(occlusion_aug)

    train_transforms = T.Compose(base_transforms)

    # 验证变换（无数据增强）
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # 创建数据集
    dataset = FACTORY[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    print(f"Dataset: {cfg.DATASETS.NAMES}")
    print(f"Num classes: {num_classes}")
    print(f"Num cameras: {cam_num}")
    print(f"Num views: {view_num}")
    print(f"Train samples: {len(train_set)}")

    # 创建训练 dataloader（使用 RandomMultipleGallerySampler，与原始 train_climb.py 一致）
    sampler = RandomMultipleGallerySampler(dataset.train, cfg.DATALOADER.NUM_INSTANCE)
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    train_loader = IterLoader(train_loader, cfg.SOLVER.ITERS if hasattr(cfg.SOLVER, 'ITERS') else None)

    # 创建验证 dataloader
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_collate_fn
    )

    # 创建 cluster loader（用于构建记忆库）
    cluster_set = ImageDataset(dataset.train, val_transforms)
    # 使用配置文件中的 CLUSTER_BATCH_SIZE，如果不存在则默认使用 2048
    cluster_batch_size = getattr(cfg.DATALOADER, 'CLUSTER_BATCH_SIZE', 2048)
    cluster_loader = DataLoader(
        cluster_set,
        batch_size=cluster_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, cluster_loader, len(dataset.query), num_classes, cam_num, view_num
