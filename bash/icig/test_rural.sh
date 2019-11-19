python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/coco_qinghai_rural/e2e_mask_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/coco_qinghai_rural/e2e_mask_rcnn_R_101_FPN_1x_gpu3_trec.yaml
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/coco_qinghai_rural/e2e_faster_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/coco_qinghai_rural/e2e_faster_rcnn_R_101_32x8d_FPN_1x_gpu3_trec.yaml
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/coco_qinghai_rural/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file configs/coco_qinghai_rural/e2e_bquad_rcnn_R_101_32x8d_FPN_1x_gpu3_trec.yaml