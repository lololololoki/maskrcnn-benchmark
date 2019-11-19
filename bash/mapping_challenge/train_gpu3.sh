#!/usr/bin/env bash
GPUS=3
CUDA_VISIBLE_DEVICES=7,8,9 python -m torch.distributed.launch \
--master_port=$((RANDOM + 10000)) \
--nproc_per_node=$GPUS tools/train_net.py \
--config-file "configs/coco_mapping_challenge/e2e_mask_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml"
CUDA_VISIBLE_DEVICES=7,8,9 python -m torch.distributed.launch \
--master_port=$((RANDOM + 10000)) \
--nproc_per_node=3 \
mytools/test_net_qinghai_suburban_X_bquad_trec_alldoublehead.py \
--config-file "configs/coco_mapping_challenge/e2e_mask_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml" \
--weight "output/coco_mapping_challenge/e2e_mask_rcnn_X_101_32x8d_FPN_1x_gpu3_trec/model_" \
| tee output/coco_mapping_challenge/e2e_mask_rcnn_X_101_32x8d_FPN_1x_gpu3_trec/test_results.txt