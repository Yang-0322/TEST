"""
CLIMB 训练脚本 with Path-based Feature Swapping
通过跨人员特征交换来增强遮挡鲁棒性，适用于Market1501和qiuxiu数据集
"""
import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import cfg
from climb.model_patch_swap import make_model_patch_swap
from climb.dataloader_patch_swap import make_dataloader_patch_swap
from climb.loss import ClusterMemoryAMP, CrossEntropyLabelSmooth, TripletLoss
from climb.optimizer import make_CLIMB_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval
from utils.meter import AverageMeter
from climb.utils import extract_image_features, save_checkpoint
import tqdm


def set_seed(seed):
    """设置随机种子以保证可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(epoch, train_loader, cluster_loader, model, optimizer, scheduler,
          xent, tri_loss, logger, cfg, device, swap_ratio=0.3):
    """训练一个 epoch"""
    model.train()

    loss_meter = AverageMeter()
    loss_meter1 = AverageMeter()  # PCL loss
    loss_meter2 = AverageMeter()  # ID loss
    loss_meter3 = AverageMeter()  # ID loss2
    loss_meter4 = AverageMeter()  # Triplet loss
    acc_meter = AverageMeter()
    acc_meter1 = AverageMeter()

    log_period = cfg.SOLVER.LOG_PERIOD

    # 创建记忆库（每个 epoch 都更新）
    from climb.utils import compute_cluster_centroids
    from climb.loss import ClusterMemoryAMP

    # 使用普通特征提取（不带路径交换）
    image_features, gt_labels = extract_image_features(model, cluster_loader, use_amp=True)
    image_features = image_features.float()
    image_features = nn.functional.normalize(image_features, dim=1)

    num_classes = len(gt_labels.unique()) - 1 if -1 in gt_labels else len(gt_labels.unique())
    logger.info(f'Epoch {epoch}: Memory has {num_classes} classes.')

    memory = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
    memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)
    logger.info(f'Epoch {epoch}: Memory bank shape = {memory.features.shape}')

    # 重新设置为训练模式
    model.train()

    # train one iteration
    num_iters = len(train_loader)
    for n_iter in range(num_iters):
        img, target, target_cam, target_view = train_loader.next()

        optimizer.zero_grad()

        img = img.to(device)
        target = target.to(device)
        target_cam = target_cam.to(device)

        if cfg.MODEL.SIE_CAMERA:
            target_cam = target_cam.to(device)
        else:
            target_cam = None
        if cfg.MODEL.SIE_VIEW:
            target_view = target_view.to(device)
        else:
            target_view = None

        # 前向传播（传递labels以启用路径交换）
        feat, logits, feat_sp, logits_sp = model(
            img,
            cam_label=target_cam,
            view_label=target_view,
            labels=target
        )

        # 计算损失
        loss1 = memory(feat, target) * cfg.MODEL.PCL_LOSS_WEIGHT
        loss_id = xent(logits, target) * cfg.MODEL.ID_LOSS_WEIGHT
        loss_id2 = xent(logits_sp, target)
        loss_tri = tri_loss(feat_sp, target)
        loss = loss1 + loss_id + loss_id2 + loss_tri

        loss.backward()
        optimizer.step()

        acc = (logits.max(1)[1] == target).float().mean()
        acc2 = (logits_sp.max(1)[1] == target).float().mean()

        loss_meter.update(loss.item(), img.shape[0])
        loss_meter1.update(loss1.item(), img.shape[0])
        loss_meter2.update(loss_id.item(), img.shape[0])
        loss_meter3.update(loss_id2.item(), img.shape[0])
        loss_meter4.update(loss_tri.item(), img.shape[0])
        acc_meter.update(acc, 1)
        acc_meter1.update(acc2, 1)

        torch.cuda.synchronize()

        if (n_iter + 1) % log_period == 0:
            lr = scheduler.get_lr()[0]
            logger.info("Epoch[{}] Iteration[{}/{}] "
                        "Loss_total: {:.3f}, "
                        "Loss1: {:.3f}, "
                        "Loss2: {:.3f}, "
                        "Loss3: {:.3f}, "
                        "Loss4: {:.3f}, "
                        "acc1: {:.3f},"
                        "acc2: {:.3f},"
                        "Lr: {:.2e}"
                        .format(epoch, (n_iter + 1), len(train_loader),
                                loss_meter.avg,
                                loss_meter1.avg,
                                loss_meter2.avg,
                                loss_meter3.avg,
                                loss_meter4.avg,
                                acc_meter.avg,
                                acc_meter1.avg,
                                lr))

    return loss_meter.avg, acc_meter.avg


def main():
    parser = argparse.ArgumentParser(description='CLIMB with Path-based Feature Swapping')
    parser.add_argument('--config_file', required=True, type=str)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--swap_ratio', type=float, default=0.3,
                        help='Ratio of features to swap between samples')
    parser.add_argument('--swap_layers', type=int, nargs='+', default=[6, 11],
                        help='Transformer layers to apply path swapping')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # 设置随机种子
    set_seed(cfg.SOLVER.SEED)

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载配置
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 添加路径交换相关配置到cfg（如果配置文件中没有）
    if not hasattr(cfg.MODEL, 'SWAP_RATIO'):
        cfg.MODEL.SWAP_RATIO = args.swap_ratio
    if not hasattr(cfg.MODEL, 'SWAP_LAYERS'):
        cfg.MODEL.SWAP_LAYERS = args.swap_layers

    # 创建输出目录
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # 创建 logger
    logger = setup_logger("CLIMB-PathSwap", cfg.OUTPUT_DIR, if_train=True)

    logger.info(f"Configuration: {args.config_file}")
    logger.info(f"Output directory: {cfg.OUTPUT_DIR}")
    logger.info(f"Use GPU: {args.gpu}")
    logger.info(f"Swap ratio: {cfg.MODEL.SWAP_RATIO}")
    logger.info(f"Swap layers: {cfg.MODEL.SWAP_LAYERS}")

    # 创建 dataloader
    logger.info("Creating dataloaders...")
    train_loader, val_loader, cluster_loader, num_query, num_classes, cam_num, view_num = make_dataloader_patch_swap(cfg)

    # 创建模型
    logger.info("Creating model...")
    model = make_model_patch_swap(
        cfg,
        num_classes,
        cam_num,
        view_num,
        swap_ratio=cfg.MODEL.SWAP_RATIO,
        swap_layers=cfg.MODEL.SWAP_LAYERS
    ).to(device)

    # 使用 DataParallel
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
    else:
        model = nn.DataParallel(model).cuda()

    # 加载预训练权重（如果有）
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    # 创建优化器
    optimizer = make_CLIMB_optimizer(cfg, model)

    # 创建学习率调度器
    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        cfg.SOLVER.WARMUP_FACTOR,
        cfg.SOLVER.WARMUP_ITERS,
        cfg.SOLVER.WARMUP_METHOD
    )

    # 创建损失函数
    xent = CrossEntropyLabelSmooth(num_classes)
    tri_loss = TripletLoss()

    logger.info(f'smoothed cross entropy loss on {num_classes} classes.')

    # 创建评估器
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger.info('start training')
    logger.info(f"Swap ratio: {cfg.MODEL.SWAP_RATIO}")
    logger.info(f"Swap layers: {cfg.MODEL.SWAP_LAYERS}")

    best_performance = 0
    best_epoch = 1

    # 训练循环
    for epoch in range(1, epochs + 1):
        loss_avg, acc_avg = train(
            epoch, train_loader, cluster_loader, model, optimizer, scheduler,
            xent, tri_loss, logger, cfg, device, swap_ratio=cfg.MODEL.SWAP_RATIO
        )

        scheduler.step()
        logger.info(f"Epoch {epoch} done. Loss: {loss_avg:.3f}, Acc: {acc_avg:.3f}")

        # 每 10 个 epoch 保存一次模型权重
        if epoch % checkpoint_period == 0:
            checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth.tar')
            save_checkpoint(model.state_dict(), False, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # 评估
        if epoch % eval_period == 0:
            logger.info(f"Starting evaluation for epoch {epoch}...")

            model.eval()
            evaluator.reset()

            for n_iter, (img, pid, camid, camid_batch, viewid, img_paths) in enumerate(tqdm.tqdm(val_loader, desc="Validating")):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camid.to(device)
                    else:
                        camids = None
                    target_view = None

                    # 验证时不使用路径交换（传递None作为labels）
                    feat, feat1, feat2 = model(img, cam_label=camids, view_label=target_view, labels=None)
                    evaluator.update((feat, feat1, feat2, pid, camid))

            cmc, mAP, cmc01, mAP01, cmc02, mAP02, cmc03, mAP03 = evaluator.compute()

            logger.info(f"Validation Results - Epoch: {epoch}")
            logger.info(f"mAP: {mAP:.1%}")
            for r in [1, 5, 10, 20]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r-1]:.1%}")

            logger.info(f"mAP_1: {mAP01:.1%}")
            for r in [1, 5, 10, 20]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc01[r-1]:.1%}")

            logger.info(f"mAP_2: {mAP02:.1%}")
            for r in [1, 5, 10, 20]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc02[r-1]:.1%}")

            logger.info(f"mAP_3: {mAP03:.1%}")
            for r in [1, 5, 10, 20]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc03[r-1]:.1%}")

            torch.cuda.empty_cache()

            prec1 = cmc[0] + mAP
            is_best = prec1 > best_performance
            best_performance = max(prec1, best_performance)
            if is_best:
                best_epoch = epoch

            save_checkpoint(model.state_dict(), is_best,
                          os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

            torch.cuda.empty_cache()

    logger.info(f"==> Best Perform {best_performance:.1%}, achieved at epoch {best_epoch}")
    logger.info('Training done.')
    print(cfg.OUTPUT_DIR)


if __name__ == '__main__':
    import numpy as np
    main()
