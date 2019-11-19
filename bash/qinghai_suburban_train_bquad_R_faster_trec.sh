python -m torch.distributed.launch --master_port=8007 --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/coco_qinghai_suburban/e2e_faster_rcnn_R_101_32x8d_FPN_1x_gpu3_trec.yaml" && python -m torch.distributed.launch --master_port=8007 --nproc_per_node=$NGPUS mytools/test_net_qinghai_suburban_R_faster_trec.py | tee output/coco_qinghai_suburban/e2e_faster_rcnn_R_101_32x8d_FPN_1x_gpu3_trec/inference/coco_qinghai_suburban_test_trec/test_results.txt