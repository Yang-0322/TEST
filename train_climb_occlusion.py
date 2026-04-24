"""
遮挡鲁棒的 CLIMB 训练脚本
专门针对篮球球员 ReID 的遮挡问题
"""

import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import cfg
from climb.model_occlusion import make_model_occlusion_robust
from climb.dataloader_occlusion import make_dataloader_with_occlusion
from climb.loss import ClusterMemoryAMP, CrossEntropyLabelSmooth, TripletLoss
from climb.optimizer import make_CLIMB_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval
from utils.meter import AverageMeter
from climb.utils import save_checkpoint
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='CLIMB with Occlusion Robustness Training')
    parser.add_argument('--config_file', required=True, type=str)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def train(epoch, train_loader, cluster_loader, model, optimizer, scheduler, xent, tri_loss, logger, cfg, device):
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
    from climb.utils import extract_image_features, compute_cluster_centroids
    from climb.processor_climb import ClusterMemoryAMP

    image_features, gt_labels = extract_image_features(model, cluster_loader, use_amp=True)
    image_features = image_features.float()
    image_features = F.normalize(image_features, dim=1)

    num_classes = len(gt_labels.unique()) - 1 if -1 in gt_labels else len(gt_labels.unique())
    logger.info(f'Epoch {epoch}: Memory has {num_classes} classes.')

    memory = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
    memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)
    logger.info(f'Epoch {epoch}: Memory bank shape = {memory.features.shape}')

    # 重新设置为训练模式（因为 extract_image_features 会设置为 eval）
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

        # 前向传播
        if cfg.MODEL.USE_OCCLUSION_AWARE:
            # 使用遮挡感知模型
            outputs = model(img, cam_label=target_cam, view_label=target_view)
            if len(outputs) == 5:
                feat, logits, feat_sp, logits_sp, occlusion_map = outputs
            else:
                feat, logits, feat_sp, logits_sp = outputs
                occlusion_map = None
        else:
            # 使用原始模型
            feat, logits, feat_sp, logits_sp = model(img, cam_label=target_cam, view_label=target_view)

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


def main():
    args = parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载配置
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 创建输出目录
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # 创建 logger
    logger = setup_logger("CLIMB", cfg.OUTPUT_DIR, if_train=True)

    logger.info(f"Configuration: {args.config_file}")
    logger.info(f"Output directory: {cfg.OUTPUT_DIR}")
    logger.info(f"Use GPU: {args.gpu}")
    logger.info(f"Occlusion aware: {cfg.MODEL.USE_OCCLUSION_AWARE}")

    # 创建 dataloader
    logger.info("Creating dataloaders...")
    train_loader, val_loader, cluster_loader, num_query, num_classes, cam_num, view_num = make_dataloader_with_occlusion(cfg)

    # 创建模型
    logger.info("Creating model...")
    model = make_model_occlusion_robust(
        cfg,
        num_classes,
        cam_num,
        view_num,
        use_occlusion_aware=cfg.MODEL.USE_OCCLUSION_AWARE
    ).to(device)

    # 使用 DataParallel
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        print('Using {} GPUs for training'.format(torch.cuda.device_count()))
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = nn.DataParallel(model).cuda()

    # 加载预训练权重（如果有）
    # if hasattr(cfg.TEST, 'PRETRAINED_PATH') and cfg.TEST.PRETRAINED_PATH:
    #     logger.info(f"Loading pretrained model from {cfg.TEST.PRETRAINED_PATH}")
    #     model.load_param_finetune(cfg.TEST.PRETRAINED_PATH)

    # 创建优化器
    optimizer = make_CLIMB_optimizer(cfg, model)

    # 创建学习率调度器
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD
    )

    # 创建损失函数
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    tri_loss = TripletLoss()
    logger.info(f'smoothed cross entropy loss on {num_classes} classes.')

    # 创建记忆库（在 train 函数中每个 epoch 重新创建）
    memory = None  # 不在这里创建，在 train 函数中每个 epoch 重新创建

    # 训练循环
    logger.info("Starting training...")
    best_performance = 0.0
    best_epoch = 1
    start_epoch = 1

    # 恢复训练（如果有 checkpoint）
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_param(args.resume)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    # 初始化评估器
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS + 1):
        train_loader.new_epoch()

        # 训练
        train(
            epoch,
            train_loader,
            cluster_loader,
            model,
            optimizer,
            lr_scheduler,
            xent,
            tri_loss,
            logger,
            cfg,
            device
        )

        lr_scheduler.step()
        logger.info("Epoch {} done.".format(epoch))

        # 每 10 个 epoch 保存一次模型权重
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f'checkpoint_epoch_{epoch}.pth.tar')
            save_checkpoint(model.state_dict(), False, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # 验证
        if epoch % cfg.SOLVER.EVAL_PERIOD == 0:
            logger.info(f"Starting evaluation for epoch {epoch}...")
            model.eval()
            evaluator.reset()

            for n_iter, (img, vid, camid, camids_batch, trackid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids_batch.to(device)
                    else:
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = trackid.to(device)
                    else:
                        target_view = None
                    # 模型在验证模式下返回3或4个值
                    outputs = model(img, cam_label=camids, view_label=target_view)
                    if len(outputs) == 4:
                        feat, feat1, feat2, occlusion_map = outputs
                    else:
                        feat, feat1, feat2 = outputs
                    evaluator.update((feat, feat1, feat2, vid, camid))

            # 计算指标
            cmc, mAP, cmc01, mAP01, cmc02, mAP02, cmc03, mAP03 = evaluator.compute()

            # 打印结果
            logger.info(f"Validation Results - Epoch: {epoch}")
            logger.info(f"mAP: {mAP:.1%}")
            for r in [1, 5, 10, 20]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

            logger.info(f"mAP_1: {mAP01:.1%}")
            for r in [1, 5, 10, 20]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc01[r - 1]:.1%}")

            logger.info(f"mAP_2: {mAP02:.1%}")
            for r in [1, 5, 10, 20]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc02[r - 1]:.1%}")

            logger.info(f"mAP_3: {mAP03:.1%}")
            for r in [1, 5, 10, 20]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc03[r - 1]:.1%}")

            # 保存最佳模型
            prec1 = cmc[0] + mAP
            is_best = prec1 > best_performance
            best_performance = max(prec1, best_performance)
            if is_best:
                best_epoch = epoch
                save_checkpoint(model.state_dict(), is_best, os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

            torch.cuda.empty_cache()
            model.train()

    logger.info("==> Best Perform {:.1%}, achieved at epoch {}".format(best_performance, best_epoch))
    logger.info('Training done.')
    print(cfg.OUTPUT_DIR)


if __name__ == '__main__':
    main()
