import os
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.coco_pansharpening import COCODataset
from maskrcnn_benchmark.data.transforms import build_transforms

def main():

    # transformer
    config_file = '/home/lqp2018/mnt/lqp2018/jky/model/maskrcnn-benchmark-8008/configs/coco_qinghai_urban/e2e_mask_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml'
    cfg.merge_from_file(config_file)
    cfg.freeze()
    transforms = build_transforms(cfg, True)

    root = '/home/lqp2018/mnt/lqp2018/jky/model/maskrcnn-benchmark-8008/datasets/coco_shandong_pansharpening/images'
    ann_file = '/home/lqp2018/mnt/lqp2018/jky/model/maskrcnn-benchmark-8008/datasets/coco_shandong_pansharpening/voc_0712_trainval_trec.json'
    remove_images_without_annotations = True
    my_coco = COCODataset(ann_file, root, remove_images_without_annotations, transforms)
    for images, target, _ in my_coco:
        image_pan = images[1]
        image_mul = images[2]
        image = torch.cat((image_mul, image_pan), dim=0)
        print (image.shape, image_pan.shape, image_mul.shape)
        print (image)
        print (image_pan.max(), image_pan.min())
        print (image_mul.max(), image_mul.min())
        print (image.max(), image.min())
        break


if __name__ == "__main__":
    main()