import json
import numpy as np
from pycocotools.coco import COCO
# import seaborn as sns
import matplotlib.pyplot as plt

# sns.set( palette="muted", color_codes=True)


# coco_trainval2017 = COCO("{}/instances_val2017.json".format("/home/jky-cuda8/data/cocotrainval2017"))
# coco_qinghai_test = COCO("{}/voc_0712_test.json".format("/home/jky-cuda8/data/qinghai"))
# coco_qinghai_train = COCO("{}/voc_0712_trainval.json".format("/home/jky-cuda8/data/qinghai"))

ROOT = "/home/lqp2018/mnt/lqp2018/jky/model/maskrcnn-benchmark-8008/datasets/coco_qinghai/"

cocos = []

coco_qinghai_rural_test = COCO("{}/voc_0712_test_trec.json".format(ROOT + "VOC0712_rural"))
coco_qinghai_rural_train = COCO("{}/voc_0712_trainval_trec.json".format(ROOT + "VOC0712_rural"))
coco_qinghai_suburban_test = COCO("{}/voc_0712_test_trec.json".format(ROOT + "VOC0712_suburban_correct"))
coco_qinghai_suburban_train = COCO("{}/voc_0712_trainval_trec.json".format(ROOT + "VOC0712_suburban_correct"))
coco_qinghai_urban_test = COCO("{}/voc_0712_test_trec.json".format(ROOT + "VOC0712_urban"))
coco_qinghai_urban_train = COCO("{}/voc_0712_trainval_trec.json".format(ROOT + "VOC0712_urban"))

cocos.extend([coco_qinghai_rural_test, coco_qinghai_rural_train,
             coco_qinghai_suburban_test, coco_qinghai_suburban_train,
             coco_qinghai_urban_test, coco_qinghai_urban_train])

areas = ["Rural", "Suburban", "Urban"]
images_sum = 0
annotations_sum = 0

for idx, coco in enumerate(areas):
    # print (idx)
    coco_test = cocos[idx * 2]
    coco_train = cocos[idx * 2 + 1]
    
    images = len(coco_test.dataset['images']) + len(coco_train.dataset['images'])
    annotations = len(coco_test.dataset['annotations']) + len(coco_train.dataset['annotations'])

    images_sum += images
    annotations_sum += annotations

    print ("{} images:{}".format(areas[idx], images))
    print ("{} annotations:{}".format(areas[idx], annotations))

print("Total images:{}".format(images_sum))
print("Total annotations:{}".format(annotations_sum))


# coco_qinghai_4points = COCO("{}/voc_0712_trainval_4points.json".format("/home/jky-cuda8/data/qinghai"))
# coco_qinghai_bquad = COCO("{}/voc_0712_test_bquad.json".format("/home/jky-cuda8/data/qinghai"))
# coco_qinghai_trec = COCO("{}/voc_0712_test_trec.json".format("/home/jky-cuda8/data/qinghai"))
# coco_qinghai_bquad_greater_than_4_rec = COCO("{}/voc_0712_test_bquad_greater_than_4_rec.json".format("/home/jky-cuda8/data/qinghai"))

# coco = COCO("{}/annotation-hasbbox-filter.json".format("/home/jky-cuda8/model/crowdai-mapping-challenge-mask-rcnn/data/train"))
# qinghai_static_points = np.zeros(10000)
# for ann in coco_qinghai.anns:
#     num_points = len(coco_qinghai.anns[ann]["segmentation"][0]) // 2
#     qinghai_static_points[num_points] += 1
#     qinghai_num_points_max = max(qinghai_num_points_max, num_points)

# image_ids = coco_qinghai_train.getImgIds()
# ids = [img_id for img_id in coco_qinghai_bquad.getImgIds() if len(coco_qinghai_bquad.getAnnIds(imgIds=img_id, iscrowd=None)) == 0]
# json.dump(coco_qinghai_4points.dataset, open('/home/jky-cuda8/model/crowdai-mapping-challenge-mask-rcnn/data/train/annotation-hasbbox-indent0.json','w'))

# areas = np.array([int(coco_trainval2017.anns[ann]['area']) for ann in coco_trainval2017.anns])
# sns.set_style('darkgrid')
# sns.distplot(areas)