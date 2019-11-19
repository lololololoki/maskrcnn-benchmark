#!/usr/bin/env bash
GPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch \
--master_port=$((RANDOM + 10000)) \
--nproc_per_node=$GPUS tools/train_net.py \
--config-file "configs/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_alldoublehead.yaml"
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch \
--master_port=$((RANDOM + 10000)) \
--nproc_per_node=$GPUS \
mytools/test_net_qinghai_suburban_X_bquad_trec_alldoublehead.py \
--config-file "configs/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_alldoublehead.yaml" \
| tee output/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu6_trec_alldoublehead/inference/coco_qinghai_suburban_correct_test_trec/test_results.txt