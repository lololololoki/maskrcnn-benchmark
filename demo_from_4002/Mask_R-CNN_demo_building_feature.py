# coding: utf-8

# # Mask R-CNN demo
#
# This notebook illustrates one possible way of using `maskrcnn_benchmark` for computing predictions on images from an arbitrary URL.
#
# Let's start with a few standard imports

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

from tqdm import tqdm

import os

# set GPU No.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12


from maskrcnn_benchmark.config import cfg
from predictor import COCODemo



config_file = "/data/lqp2018/jky/model/maskrcnn-benchmark-8008/configs/" \
              "coco_qinghai_urban/" \
              "e2e_bquad_rcnn_X_101_32x8d_FPN_1x_gpu3_trec.yaml"
cfg.merge_from_file(config_file)

coco_demo = COCODemo(
    cfg,
    min_image_size=512,
    confidence_threshold=0.5,
)

coco_demo.CATEGORIES = [
        "__background",
        "building",
    ]


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


# Let's now load an image from the COCO dataset. It's reference is in the comment

# In[7]:

import cv2
# from http://cocodataset.org/#explore?id=345434
# image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
image = cv2.imread('/home/jky-cuda8/data/ship_2class/images/000003.jpg')
# imshow(image)


# ### Computing the predictions
#
# We provide a `run_on_opencv_image` function, which takes an image as it was loaded by OpenCV (in `BGR` format), and computes the predictions on them, returning an image with the predictions overlayed on the image.

# In[8]:


# compute predictions


def showImage(im, index, OutDir):
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
    savename = os.path.join(OutDir, item[:-4]) + '_P' + str(index) + '.png'
    print(savename)
    plt.set_cmap('jet')
    plt.subplots_adjust(top=0.975, bottom=0.025, left=0.025, right=0.975, hspace=0, wspace=0)
    fig = plt.gcf()
    fig.set_size_inches(20.48, 20.48)
    plt.imshow(im)
    fig.savefig(savename)
    plt.close

def visFeature(features, OutDir):
    """
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
        f = np.squeeze(f).mean(0)
        # use sigmod to [0,1]
        # f = 1.0 / (1 + np.exp(-1 * f))
        # # to [0,255]
        # f = np.round(f * 255)
        showImage(f, i + 2, OutDir)
        # cv2.imwrite(os.path.join(OutDir, 'demo_P%d.png'%(i+2)), f)

OutDir = '/data/lqp2018/jky/model/maskrcnn-benchmark-8008/demo_from_4002/output/building_canada_mapping'
for root, dirs, files in tqdm(os.walk(OutDir)):
    for item in tqdm(files):
        if item.endswith('.jpg'):
            image = cv2.imread(os.path.join(OutDir, item))
            predictions, features = coco_demo.run_on_opencv_image(image)
            visFeature(features, OutDir)









# imshow(predictions)


# # ## Keypoints Demo
#
# # In[9]:
#
#
# # set up demo for keypoints
# config_file = "G:\model\20190317_maskrcnn_benchmark_4002\configs\coco_ship_2class\e2e_faster_rcnn_X_101_32x8d_FPN_gpu1.yaml"
# cfg.merge_from_file(config_file)
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
# cfg.merge_from_list(["MODEL.MASK_ON", False])
#
# coco_demo = COCODemo(
#     cfg,
#     min_image_size=800,
#     confidence_threshold=0.7,
# )
#
#
# # In[10]:
#
# import cv2
#
# # run demo
# # image = load("http://farm9.staticflickr.com/8419/8710147224_ff637cc4fc_z.jpg")
# image = cv2.imread('/home/jky-cuda8/data/ship_2class/images/000003.jpg')
# predictions = coco_demo.run_on_opencv_image(image)
# imshow(predictions)

