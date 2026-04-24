"""
CLIMB 路径交换模型评估脚本
"""
import argparse
import logging
import os
import torch
import torch.nn as nn
import tqdm
from config import cfg
from climb.model_patch_swap import make_model_patch_swap
from climb.dataloader_patch_swap import make_dataloader_patch_swap
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval


def do_inference(cfg, model, val_loader, num_query, device):
    """推理"""
    logger = logging.getLogger("CLIMB-PathSwap")
    logger.info("Enter inferencing")

    model.to(device)
    model.eval()

    evaluator = R1_mAP_eval(num_query, max_rank=50,
                          feat_norm=cfg.TEST.FEAT_NORM,
                          reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    logger.info(f"Starting evaluation ...")

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

    logger.info("Validation Results")
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

    return cmc[0], mAP


def main():
    parser = argparse.ArgumentParser(description='CLIMB Path Swap Evaluation')
    parser.add_argument('--config_file', required=True, type=str)
    parser.add_argument('--resume', type=str, default=None, help='Path to the trained model checkpoint')
    parser.add_argument('--swap_ratio', type=float, default=0.3,
                        help='Ratio of features to swap between samples')
    parser.add_argument('--swap_layers', type=int, nargs='+', default=[6, 11],
                        help='Transformer layers to apply path swapping')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # 设置 GPU
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

    # 创建 logger
    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("CLIMB-PathSwap", output_dir, if_train=False)

    logger.info(f"Configuration: {args.config_file}")
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
        print('Using {} GPUs for evaluation'.format(torch.cuda.device_count()))
    else:
        model = nn.DataParallel(model).cuda()

    # 加载模型权重
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    elif cfg.TEST.PRETRAINED_PATH and cfg.TEST.PRETRAINED_PATH != "pretrain_model_path":
        logger.info(f"Loading checkpoint from {cfg.TEST.PRETRAINED_PATH}")
        checkpoint = torch.load(cfg.TEST.PRETRAINED_PATH, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.warning("No checkpoint specified. Using random initialization.")

    # 评估
    rank1, mAP = do_inference(cfg, model, val_loader, num_query, device)

    logger.info(f"Final Results - Rank-1: {rank1:.1%}, mAP: {mAP:.1%}")


if __name__ == '__main__':
    main()
