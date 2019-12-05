
# coding: utf-8

# # Mask R-CNN demo
# 
# This notebook illustrates one possible way of using `maskrcnn_benchmark` for computing predictions on images from an arbitrary URL.
# 
# Let's start with a few standard imports

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np


# In[2]:


# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12


# Those are the relevant imports for the detection model

# In[3]:


from maskrcnn_benchmark.config import cfg
from predictor import COCODemo


# We provide a helper class `COCODemo`, which loads a model from the config file, and performs pre-processing, model prediction and post-processing for us.
# 
# We can configure several model options by overriding the config options.
# In here, we make the model run on the CPU

# In[4]:


config_file = "/home/jky-cuda8/model/20190305_maskrcnn_benchmark/maskrcnn-benchmark/configs/" \
              "coco_dlr_pkw_1class/" \
              "e2e_faster_rcnn_X_101_32x8d_FPN_gpu1.yaml"
cfg.merge_from_file(config_file)

coco_demo = COCODemo(
    cfg,
    min_image_size=512,
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


# Let's define a few helper functions for loading images from a URL

# In[6]:


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
import os
from PIL import Image


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
    savename = os.path.join(OutDir, item[:-4]) + '_P' + str(index) + '.pdf'
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

OutDir = '/home/jky-cuda8/model/20190305_maskrcnn_benchmark/maskrcnn-benchmark/demo/output/DLR'
for root, dirs, files in os.walk(OutDir):
    for item in files:
        if item.endswith('.png'):
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

