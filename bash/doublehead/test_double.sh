python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/coco_qinghai_suburban_correct/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_alldoublehead.yaml