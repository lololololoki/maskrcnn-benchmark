CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((RANDOM + 10000)) \
    mytools/test_net_bquad.py \
    --config-file configs/coco_shandong_pansharpening/e2e_PanMulImg_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml