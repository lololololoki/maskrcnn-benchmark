python -m torch.distributed.launch --master_port=8008 --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/coco_fujian_X-101_512/e2e_bquad_rcnn_X_101_32x8d_FPN_atbox_atbquad_1x_gpu3_trec.yaml" && python -m torch.distributed.launch --master_port=8008 --nproc_per_node=$NGPUS mytools/test_net_fujian_bquad_atbox_atbquad.py | tee /home/lqp2018/mnt/lqp2018/jky/model/maskrcnn-benchmark-8008/output/fujian_X-101_512/e2e_bquad_rcnn_X_101_32x8d_FPN_atbox_atbquad_1x_gpu3_trec/inference/coco_fujian_test_trec/test_results.txt