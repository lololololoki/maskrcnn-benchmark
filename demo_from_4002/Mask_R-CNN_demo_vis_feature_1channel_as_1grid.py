# coding: utf-8

# # Mask R-CNN demo
# 
# This notebook illustrates one possible way of using `maskrcnn_benchmark` for computing predictions on images from an arbitrary URL.
# 
# Let's start with a few standard imports

# 2019年3月23日 14点21分 jky
# 用pycharm执行前记得修改Working directory

import matplotlib.pyplot as plt
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import cv2
import os

def showImage(im, index, OutDir, size, num):
    """
    Arguments:
        im (array[int]): image.
        index (int): index of the image.
        OutDir (str): output path
    Returns:
        void
    """
    if im.ndim == 3:
        im = im[:, :, ::-1]
    savename = os.path.join(OutDir, item[:-4]) + '_P' + str(index) + '.pdf'
    print(savename)
    plt.set_cmap('jet')

    # 去掉坐标
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=0.999, bottom=0.001, left=0.001, right=0.999, hspace=0, wspace=0)
    plt.subplots_adjust(top=0.975, bottom=0.025, left=0.025, right=0.975, hspace=0, wspace=0)
    # plt.margins(0, 0)

    fig = plt.gcf()
    fig.set_size_inches(20.48, 20.48)
    new_ticks = np.linspace(0, size, num + 1)
    plt.xticks(new_ticks)
    plt.yticks(new_ticks)
    # plt.grid(True, color='k')
    plt.imshow(im)

    # 绘制水平以及垂直分割线
    piece = size // num
    # plt.axhline(7.5)
    # plt.axvline(7.5)
    # plt.axhline(120-0.5)
    # plt.axvline(120-0.5)
    for p in range(piece, size, piece):
        # print(p)
        plt.axhline(p-0.5)
        plt.axvline(p-0.5)

    fig.savefig(savename)
    # plt.ion()
    # plt.pause(0.1)
    plt.clf()

def visFeature(features, OutDir):
    """
    分别将p2 p3 p4 p5 p6上每个通道的特张图拼接后绘制
    Arguments:
        features (list[array[]]): feature maps for each feature level.
            where array[] = array[P2, P3, P4, P5], and
            P2 (1, 256, 128, 128)
            P3 (1, 256, 64, 64)
            P4 (1, 256, 32, 32)
            P5 (1, 256, 16, 16)
            P6 (1, 256, 8, 8)
        OutDir (str): output path
    Returns:
        void
    """
    for i, f in enumerate(features):
        print(i)
        # if i != 4:
        #     continue
        f = np.squeeze(f)
        raws = cols = int(f.shape[0] ** 0.5)

        xPiece = f.shape[1]
        yPiece = f.shape[2]

        xSize = int(raws * xPiece)
        ySize = int(cols * yPiece)

        f_grid_image = np.zeros((xSize, ySize))
        for raw in range(raws):
            for col in range(cols):
                f_index = raw * cols + col
                f_grid_image[raw*xPiece:raw*xPiece+xPiece,
                             col*yPiece:col*yPiece+yPiece] = f[f_index]

        showImage(f_grid_image, i + 2, OutDir, xSize, raws)
        # cv2.imwrite(os.path.join(OutDir, 'demo_P%d.png'%(i+2)), f)

config_file = "/home/jky-cuda8/model/20190305_maskrcnn_benchmark/maskrcnn-benchmark/configs/" \
              "coco_dlr_pkw_1class/" \
              "e2e_faster_rcnn_X_101_32x8d_FPN_gpu1.yaml"
cfg.merge_from_file(config_file)

coco_demo = COCODemo(
    cfg,
    min_image_size=256,
    confidence_threshold=0.5,
)

coco_demo.CATEGORIES = [
        "__background",
        "pkw",
        # "person",
        # "bicycle",
        # "car",
        # "motorcycle",
        # "airplane",
        # "bus",
        # "train",
        # "truck",
        # "boat",
        # "traffic light",
        # "fire hydrant",
        # "stop sign",
        # "parking meter",
        # "bench",
        # "bird",
        # "cat",
        # "dog",
        # "horse",
        # "sheep",
        # "cow",
        # "elephant",
        # "bear",
        # "zebra",
        # "giraffe",
        # "backpack",
        # "umbrella",
        # "handbag",
        # "tie",
        # "suitcase",
        # "frisbee",
        # "skis",
        # "snowboard",
        # "sports ball",
        # "kite",
        # "baseball bat",
        # "baseball glove",
        # "skateboard",
        # "surfboard",
        # "tennis racket",
        # "bottle",
        # "wine glass",
        # "cup",
        # "fork",
        # "knife",
        # "spoon",
        # "bowl",
        # "banana",
        # "apple",
        # "sandwich",
        # "orange",
        # "broccoli",
        # "carrot",
        # "hot dog",
        # "pizza",
        # "donut",
        # "cake",
        # "chair",
        # "couch",
        # "potted plant",
        # "bed",
        # "dining table",
        # "toilet",
        # "tv",
        # "laptop",
        # "mouse",
        # "remote",
        # "keyboard",
        # "cell phone",
        # "microwave",
        # "oven",
        # "toaster",
        # "sink",
        # "refrigerator",
        # "book",
        # "clock",
        # "vase",
        # "scissors",
        # "teddy bear",
        # "hair drier",
        # "toothbrush",
    ]

OutDir = '/home/jky-cuda8/model/20190305_maskrcnn_benchmark/maskrcnn-benchmark/demo/output/DLR'
for root, dirs, files in os.walk(OutDir):
    for item in files:
        if item.endswith('.png'):
            image = cv2.imread(os.path.join(OutDir, item))
            predictions, features = coco_demo.run_on_opencv_image(image)
            visFeature(features, OutDir)
            cv2.imwrite(os.path.join(OutDir, item[:-4] + '_result.png'), predictions)


