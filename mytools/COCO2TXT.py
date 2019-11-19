# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import cv2
import numpy as np

import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        out = []
        for id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=id)

            for ann in self.coco.loadAnns(ann_ids):
                if len(ann['segmentation']) > 0 and len(ann['bbox']) > 0:
                       # and ann['area'] > 10 and ann['area'] < 20000:  # filter area > 10 and < 20000
                    out.append(id)
                    break

        self.ids = out

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        ###jky###
        self.transforms = transforms
        # self.transforms = None
    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]

        scores = [obj["score"] for obj in anno]

        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        # guard against no boxes
        if not boxes:
            raise ValueError("Image id {} ({}) doesn't have boxes annotations!".format(self.ids[idx], anno))

        return file_name, img, boxes, scores, idx, anno

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def loadResult(self, resultFile):
        self.coco = self.coco.loadRes(resultFile)


def COCO2TXT(area, backbone):
    root = '/home/lqp2018/mnt/lqp2018/jky/model/maskrcnn-benchmark-8008/datasets/coco_qinghai/VOC0712_' + area
    imgpath = os.path.join(root, 'images')
    ann_file = os.path.join(root, 'voc_0712_test_trec.json')

    outDir = os.path.join('/home/lqp2018/mnt/lqp2018/jky/model/maskrcnn-benchmark-8008/output/',
                          'coco_qinghai_' + area,
                          backbone,
                          'inference',
                          'coco_qinghai_' + area + '_test_trec')
    resultFile = os.path.join(outDir,
                              'bbox.json')
    remove_images_without_annotations = False

    # OUTPUT = os.path.join(root, 'vis')
    outFile = open(os.path.join('/home/lqp2018/mnt/lqp2018/jky/model/maskrcnn-benchmark-8008/output/txt',
                           'det_' + area + '_' + backbone + '_.txt'),
                   'w')
    dataset = COCODataset(ann_file, imgpath, remove_images_without_annotations)
    dataset.loadResult(resultFile)

    for file_name, img, boxes, scores, idx, anno in dataset:
        print(os.path.join(root, file_name))
        fileName = file_name[0:-4]
        img = np.array(img)
        for box, score in zip(boxes, scores):
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            outFile.write("{} {} {} {} {} {}\n".format(fileName, score, x1, y1, x2, y2))
    outFile.close()

areas = ['rural', 'suburban_correct', 'urban']
areas = ['rural', 'urban']
backbones = ['e2e_mask_rcnn_X_101_32x8d_FPN_1x_gpu3_trec', 'e2e_mask_rcnn_R_101_FPN_1x_gpu3_trec',
             'e2e_faster_rcnn_X_101_32x8d_FPN_1x_gpu3_trec', 'e2e_faster_rcnn_R_101_32x8d_FPN_1x_gpu3_trec',
             'e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec', 'e2e_bquad_rcnn_R_101_32x8d_FPN_1x_gpu3_trec']
# backbones = ['e2e_faster_rcnn_R_101_FPN_gpu2', 'e2e_faster_rcnn_R_50_C4_gpu2',
#              'e2e_faster_rcnn_R_50_FPN_gpu2', 'e2e_faster_rcnn_X_101_32x8d_FPN_gpu2',
#              'e2e_mask_rcnn_R_101_FPN_gpu2', 'e2e_mask_rcnn_R_50_C4_gpu2',
#              'e2e_mask_rcnn_R_50_FPN_gpu2', 'e2e_mask_rcnn_X_101_32x8d_FPN_gpu2',
#              'retinanet_R-101-FPN_gpu2', 'retinanet_R-50-FPN_gpu2',
#              'retinanet_X_101_32x8d_FPN_gpu2']
for area in areas:
    for backbone in backbones:
        COCO2TXT(area, backbone)
# backbone = ''
# COCO2TXT('urban', backbone)

