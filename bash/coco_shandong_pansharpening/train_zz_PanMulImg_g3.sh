CUDA_VISIBLE_DEVICES=0,1,7 python -m torch.distributed.launch --master_port=$((RANDOM + 10000)) --nproc_per_node=3 tools/train_net.py --config-file "configs/coco_shandong_pansharpening/e2e_zz_PanMulImg_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml"