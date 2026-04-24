"""
CLIMB dataloader with Path-based Feature Swapping support
"""
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .dataset import ImageDataset, IterLoader
from .sampler import RandomIdentitySampler, RandomMultipleGallerySampler
from datasets.market1501 import Market1501
from datasets.qiuxiu import Qiuxiu
from datasets.msmt17 import MSMT17


FACTORY = {
    'market1501': Market1501,
    'qiuxiu': Qiuxiu,
    'msmt17': MSMT17,
}


def train_collate_fn(batch):
    """
    训练时的 collate 函数，返回数据以支持特征交换
    """
    if len(batch[0]) == 5:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
    elif len(batch[0]) == 4:
        imgs, pids, camids, viewids = zip(*batch)
        img_paths = None
    else:
        raise ValueError(f"Unexpected number of values in batch: {len(batch[0])}")

    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids


def val_collate_fn(batch):
    """
    验证时的 collate 函数
    """
    if len(batch[0]) == 5:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
    elif len(batch[0]) == 4:
        imgs, pids, camids, viewids = zip(*batch)
        img_paths = None
    else:
        raise ValueError(f"Unexpected number of values in batch: {len(batch[0])}")

    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    camids_batch = camids
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def make_dataloader_patch_swap(cfg):
    """
    创建支持路径交换的 dataloader

    Args:
        cfg: 配置对象

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        cluster_loader: 聚类数据加载器
        num_query: query数量
        num_classes: 类别数
        cam_num: 相机数量
        view_num: 视角数量
    """
    # 训练变换（包含数据增强）
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        # 使用标准的RandomErasing，让模型通过路径交换机制学习遮挡鲁棒性
        T.RandomErasing(p=cfg.INPUT.RE_PROB, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

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
        train_set,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True
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
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, cluster_loader, len(dataset.query), num_classes, cam_num, view_num
