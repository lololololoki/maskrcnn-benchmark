# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference, myinference
from maskrcnn_benchmark.modeling.detector import build_detection_model
# from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from mytools.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        # default="configs/coco_fujian_X-101/e2e_faster_rcnn_X_101_32x8d_FPN_1x_gpu3_bquad.yaml",
        default="configs/coco_qinghai_lr0.002_512/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_bquad_ICSTN_3.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--weight",
        default="output/qinghai_lr0.002_512_STEP0_3W/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_bquad_ICSTN_3/model_",
        metavar="FILE",
        help="path to weight file",
    )
    parser.add_argument(
        "--begin",
        default=500,
        metavar=int,
        help="weight begin",
    )
    parser.add_argument(
        "--end",
        default=14500,
        metavar=int,
        help="weight end",
    )
    parser.add_argument(
        "--step",
        default=500,
        metavar=int,
        help="weight step",
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR, logger=logger)

    for weight in range(args.begin, args.end+1, args.step):

        if weight == args.end:
            weight = weight - 1

        model_weight = args.weight + "%07d"%weight + ".pth"

        # print (model_weight)
        #
        # continue

        _ = checkpointer.load(model_weight)

        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON or cfg.MODEL.BQUAD_ON:
            iou_types = iou_types + ("segm",)
        output_folders = [None] * len(cfg.DATASETS.TEST)
        if cfg.OUTPUT_DIR:
            dataset_names = cfg.DATASETS.TEST
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
            
            try:
                myinference(
                    model,
                    data_loader_val,
                    iou_types=iou_types,
                    box_only=cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=output_folder,
                    cfg=cfg,
                )
                synchronize()
            except:
                pass
            # myinference(
                # model,
                # data_loader_val,
                # iou_types=iou_types,
                # box_only=cfg.MODEL.RPN_ONLY,
                # device=cfg.MODEL.DEVICE,
                # expected_results=cfg.TEST.EXPECTED_RESULTS,
                # expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                # output_folder=output_folder,
                # cfg=cfg,
            # )
            # synchronize()

if __name__ == "__main__":
    main()
