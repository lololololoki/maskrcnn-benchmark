CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((RANDOM + 10000)) \
    mytools/test_net_bquad.py \
    --config-file configs/coco_shandong_pansharpening/e2e_PanMulImg_bquad_8_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml

CUDA_VISIBLE_DEVICES=1,6,7 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    mytools/test_net_bquad.py \
    --config-file configs/coco_shandong_pansharpening/e2e_PanMulImg_bquad_8_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml \
    | tee e2e_PanMulImg_bquad_8_rcnn_X_101_32x8d_FPN_1x_gpu3_trec_results.txt

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    mytools/test_net_bquad.py \
    --config-file configs/coco_shandong_pansharpening/e2e_PanMulImg_bquad_rcnn_X_101_32x8d_FPN_1x_gpu1_trec.yaml \
    | tee e2e_PanMulImg_bquad_rcnn_X_101_32x8d_FPN_1x_gpu1_trec_results.txt

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((RANDOM + 10000)) \
    mytools/test_net_bquad.py \
    --config-file configs/coco_shandong_pansharpening/e2e_PanMulImg_bquad_rcnn_X_101_32x8d_FPN_1x_gpu1_trec.yaml \
    | tee e2e_PanMulImg_bquad_rcnn_X_101_32x8d_FPN_1x_gpu1_trec_results.txt

CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=$((RANDOM + 10000)) \
    mytools/test_net_bquad.py \
    --config-file configs/coco_shandong_pansharpening/e2e_PanMulImgPansharpening_bquad_rcnn_X_101_32x8d_FPN_1x_gpu1_trec.yaml \
    | tee results/e2e_PanMulImgPansharpening_bquad_rcnn_X_101_32x8d_FPN_1x_gpu1_trec_results.txt