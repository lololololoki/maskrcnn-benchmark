#!/usr/bin/env bash
GPUS=6
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python -m torch.distributed.launch --master_port=$((RANDOM + 10000)) \
--nproc_per_node=$GPUS tools/train_net.py \
--config-file "configs/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu6_trec_alldoublehead.yaml"
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python -m torch.distributed.launch \
--master_port=$((RANDOM + 10000)) \
--nproc_per_node=6 \
mytools/test_net_qinghai_suburban_X_bquad_trec_alldoublehead.py \
--config-file "configs/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu6_trec_alldoublehead.yaml" \
--begin=100 \
| tee output/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu6_trec_alldoublehead/test_results.txt