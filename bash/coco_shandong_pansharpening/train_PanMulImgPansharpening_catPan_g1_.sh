CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=$((RANDOM + 10000)) --nproc_per_node=1 tools/train_net.py --config-file "configs/coco_shandong_pansharpening/e2e_PanMulImgPansharpeningCatPan_bquad_rcnn_R_101_FPN_1x_gpu1_trec.yaml"
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=$((RANDOM + 10000)) --nproc_per_node=1 tools/train_net.py --config-file "configs/coco_shandong_pansharpening/e2e_zz_PanMulImg_bquad_rcnn_X_101_32x8d_FPN_1x_gpu1_trec.yaml"