"""
测试 train_climb_occlusion.py 训练出来的模型
"""

import os
import sys
import argparse

# 必须包含 ops 目录，因为 selective_scan_interface 通常在那
paths = [
    "/home/xxxxxl/python/baseline/ReID/CLIMB-ReID_version2/mamba",
    "/home/xxxxxl/python/baseline/ReID/CLIMB-ReID_version2/mamba/mamba_ssm/ops"
]

for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import numpy as np
from config import cfg
from utils.logger import setup_logger
from utils.metrics import R1_mAP_eval
from climb.model_occlusion import make_model_occlusion_robust
from climb.dataloader_occlusion import make_dataloader_with_occlusion
from tqdm import tqdm


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def do_inference(cfg, model, val_loader, num_query, logger):
    """
    执行推理并评估模型性能

    Args:
        cfg: 配置对象
        model: 模型
        val_loader: 验证数据加载器
        num_query: 查询图片数量
        logger: 日志记录器
    """
    device = "cuda"
    logger.info("Enter inferencing")
    model.to(device)
    model.eval()

    evaluator = R1_mAP_eval(
        num_query=num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING
    )
    evaluator.reset()

    logger.info(f"Starting evaluation ...")

    with torch.no_grad():
        for n_iter, (img, pid, camid, camids_batch, trackid, _) in enumerate(
            tqdm(val_loader, desc="Validating")
        ):
            img = img.to(device)

            # 处理相机标签
            if cfg.MODEL.SIE_CAMERA:
                camids = camids_batch.to(device)
            else:
                camids = None

            # 处理视角标签
            if cfg.MODEL.SIE_VIEW:
                target_view = trackid.to(device)
            else:
                target_view = None

            # 模型前向传播
            # 注意：模型在验证模式下可能返回 3 个或 4 个值
            outputs = model(img, cam_label=camids, view_label=target_view)

            if len(outputs) == 4:
                # 包含 occlusion_map 的情况
                feat_concat, out_feat, feat_sp, occlusion_map = outputs
                evaluator.update((feat_concat, out_feat, feat_sp, pid, camid))
            else:
                # 不包含 occlusion_map 的情况
                feat_concat, out_feat, feat_sp = outputs
                evaluator.update((feat_concat, out_feat, feat_sp, pid, camid))

    # 计算指标
    cmc, mAP, cmc01, mAP01, cmc02, mAP02, cmc03, mAP03 = evaluator.compute()

    # 打印结果
    logger.info("Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    logger.info("mAP_1: {:.1%}".format(mAP01))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc01[r - 1]))

    logger.info("mAP_2: {:.1%}".format(mAP02))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc02[r - 1]))

    logger.info("mAP_3: {:.1%}".format(mAP03))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc03[r - 1]))

    return cmc[0], mAP


def main():
    parser = argparse.ArgumentParser(description="Test CLIMB Occlusion Model")
    parser.add_argument(
        "--config_file",
        required=True,
        type=str,
        help="path to config file"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    # 设置 GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 加载配置
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 设置随机种子
    set_seed(cfg.SOLVER.SEED)

    # 创建输出目录
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建 logger
    logger = setup_logger("CLIMB_Occlusion_Test", output_dir, if_train=False)

    logger.info(f"Config file: {args.config_file}")
    logger.info(f"Output directory: {cfg.OUTPUT_DIR}")

    if args.config_file != "":
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info(f"Running with config:\n{cfg}")

    # 创建 dataloader
    logger.info("Creating dataloaders...")
    _, val_loader, _, num_query, num_classes, camera_num, view_num = make_dataloader_with_occlusion(cfg)

    logger.info(f"Dataset: {cfg.DATASETS.NAMES}")
    logger.info(f"Number of query images: {num_query}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Number of cameras: {camera_num}")
    logger.info(f"Number of views: {view_num}")

    # 创建模型
    logger.info("Creating model...")
    model = make_model_occlusion_robust(
        cfg,
        num_classes,
        camera_num,
        view_num,
        use_occlusion_aware=cfg.MODEL.USE_OCCLUSION_AWARE
    )

    # 使用 DataParallel
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    # 加载训练好的模型权重
    logger.info(f"Loading trained model from {cfg.TEST.PRETRAINED_PATH}")
    model.load_param(cfg.TEST.PRETRAINED_PATH)

    # 执行推理和评估
    rank1, mAP = do_inference(cfg, model, val_loader, num_query, logger)

    logger.info(f"Final Results: Rank-1={rank1:.1%}, mAP={mAP:.1%}")
    logger.info("Testing done.")


if __name__ == "__main__":
    main()