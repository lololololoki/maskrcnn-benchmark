#!/usr/bin/env bash
GPUS=3
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch \
--master_port=$((RANDOM + 10000)) \
--nproc_per_node=$GPUS tools/train_net.py \
--config-file "configs/coco_qinghai_suburban_correct/doublehead/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_tqrlossweights0.1_alldoublehead.yaml"
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch \
--master_port=$((RANDOM + 10000)) \
--nproc_per_node=3 \
mytools/test_net_qinghai_suburban_X_bquad_trec_alldoublehead.py \
--config-file "configs/coco_qinghai_suburban_correct/doublehead/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_tqrlossweights0.1_alldoublehead.yaml" \
--weight "output/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_tqrlossweights0.1_alldoublehead/model_" \
| tee output/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_tqrlossweights0.1_alldoublehead/test_results.txt