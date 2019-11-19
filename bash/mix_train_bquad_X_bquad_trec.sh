python -m torch.distributed.launch --master_port=8008 --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/coco_mix/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml" && python -m torch.distributed.launch --master_port=8008 --nproc_per_node=$NGPUS mytools/test_net_mix_X_bquad_trec.py | tee output/coco_mix/e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec/inference/test_results.txt