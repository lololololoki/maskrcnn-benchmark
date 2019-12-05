import json
import numpy as np
from pycocotools.coco import COCO
# import seaborn as sns
import matplotlib.pyplot as plt

# sns.set( palette="muted", color_codes=True)

ROOT = "/data/lqp2018/jky/model/maskrcnn-benchmark-8008/datasets/coco_shandong_pansharpening/pre_misalign/"

cocos = []

coco_shangdong_pansharpening_test = COCO("{}/voc_0712_test_trec.json".format(ROOT))
coco_shangdong_pansharpening_train = COCO("{}/voc_0712_trainval_trec.json".format(ROOT))

cocos.extend([
    coco_shangdong_pansharpening_test, coco_shangdong_pansharpening_train
])

areas = ["shangdong_pansharpening"]
images_sum = 0
annotations_sum = 0
max_side = 0

for idx, coco in enumerate(areas):
    # print (idx)
    coco_test = cocos[idx * 2]
    coco_train = cocos[idx * 2 + 1]
    
    images = len(coco_test.dataset['images'])
    annotations = len(coco_test.dataset['annotations'])

    for ann in coco_test.dataset['annotations']:
        max_side = max(max_side, ann['bbox'][2], ann['bbox'][3])

    images_sum += images
    annotations_sum += annotations
    print ("test: ")
    print ("{} images:{}".format(areas[idx], images))
    print ("{} annotations:{}".format(areas[idx], annotations))

    images = len(coco_train.dataset['images'])
    annotations = len(coco_train.dataset['annotations'])

    for ann in coco_train.dataset['annotations']:
        max_side = max(max_side, ann['bbox'][2], ann['bbox'][3])

    images_sum += images
    annotations_sum += annotations
    print("train: ")
    print("{} images:{}".format(areas[idx], images))
    print("{} annotations:{}".format(areas[idx], annotations))

print("Total images:{}".format(images_sum))
print("Total annotations:{}".format(annotations_sum))
print("Max Side:{}".format(max_side))