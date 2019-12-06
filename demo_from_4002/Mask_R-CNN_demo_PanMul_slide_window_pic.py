# coding: utf-8

# # Mask R-CNN demo
#
# This notebook illustrates one possible way of using `maskrcnn_benchmark` for computing predictions on images from an arbitrary URL.
#
# Let's start with a few standard imports

import gc
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

from tqdm import tqdm
from IPython import embed

import os

# set GPU No.
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

import torch
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, cat_boxlist


config_file = "/data/lqp2018/jky/model/maskrcnn-benchmark-8008/configs/" \
              "coco_shandong_pansharpening/" \
              "e2e_PanMulImg_bquad_rcnn_X_101_32x8d_FPN_1x_gpu1_trec.yaml"
cfg.merge_from_file(config_file)

coco_demo = COCODemo(
    cfg,
    min_image_size=512,
    confidence_threshold=0.7,
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
# image = cv2.imread('/home/jky-cuda8/data/ship_2class/images/000003.jpg')
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


def slide_window_det(image, piece_size=(512, 512), overlap=100):
    """

    :param image:
    :param piece_size:
    :param overlap:
    :return:
    """
    images_grid = []
    offsets = []
    piece_h, piece_w = piece_size
    h_size, w_size = image.shape[0:2]
    print(piece_w, piece_h)
    print(w_size, h_size)
    img_num = 0
    h_str = piece_h - overlap
    w_str = piece_w - overlap
    for h in range(0, h_size, h_str):
        for w in range(0, w_size, w_str):
            img_num += 1
            img = image[h:h + piece_h, w:w + piece_w, ]
            # padding
            if img.shape[0] < piece_h or img.shape[1] < piece_w:
                temp_img = np.zeros((piece_h, piece_w, img.shape[2]), dtype=np.uint8)
                temp_img[0:img.shape[0], 0:img.shape[1], ] = img
                img = temp_img
            offset = [h, w]
            images_grid.append(img)
            offsets.append(offset)
    return images_grid, offsets, img_num


def is_image_file(filename, form=['.tif']):
    return any(filename.endswith(extension) for extension in form)


def load_image(filepath):
    img = Image.open(filepath).convert("RGB")
    img = np.array(img)[:, :, [2, 1, 0]]
    return img


def load_PanMul_image(filepath):
    img_pan = np.array(Image.open(filepath.replace("_mul_upsample.tif", ".tif")))
    img_pan = np.expand_dims(np.array(img_pan), 2)
    img_mul_upsample = np.array(Image.open(filepath))
    img = np.concatenate((img_mul_upsample, img_pan), axis=2)
    return img


def load_PanMul_image_save_mul_upsample(filepath):
    img_pan = np.array(Image.open(filepath.replace("_mul", "")))
    img_pan = np.expand_dims(np.array(img_pan), 2)
    img_mul = np.array(Image.open(filepath))
    img_mul = cv2.pyrUp(img_mul, dstsize=(img_mul.shape[1] * 2, img_mul.shape[0] * 2))
    img_mul_upsample = cv2.pyrUp(img_mul, dstsize=(img_mul.shape[1] * 2, img_mul.shape[0] * 2))
    img_mul_upsample = cv2.resize(img_mul_upsample, (img_pan.shape[1], img_pan.shape[0]), interpolation=cv2.INTER_AREA)
    img = np.concatenate((img_mul_upsample, img_pan), axis=2)
    img_mul_upsample = Image.fromarray(img_mul_upsample[:, :, 0:4])
    # cv2.imwrite(filepath.replace("_mul.tif", "_mul_upsample.jpg"), img_mul_upsample[:, :, 0:3])
    img_mul_upsample.save(filepath.replace("_mul.tif", "_mul_upsample.tif"))
    return img


def tif2jpg(image_path):
    image_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if is_image_file(x)]
    image_list.sort()
    for image in tqdm(image_list):
        img = cv2.imread(image)
        print (image, np.array(img).shape)
        print (os.path.join(image_path.replace('pic', 'jpg')))
        cv2.imwrite(image.replace('pic', 'jpg').replace('tif', 'jpg'), img)


def dets_nms(boxlist, nms, score_field="scores"):
    boxlist_for_class = boxlist_nms(
        boxlist, nms, score_field="scores"
    )


def main():
    images_path = '/data/lqp2018/jky/model/maskrcnn-benchmark-8008/demo_from_4002/output/shangdong_pansharpening/tif_for_draw'
    image_path_list = [os.path.join(images_path, x) for x in os.listdir(images_path) if is_image_file(x, ['_mul_upsample.tif'])]
    image_path_list.sort()
    print (image_path_list)
    for image_path in tqdm(image_path_list):
        # image = cv2.imread(image_path)
        predictions_list = []
        image = load_PanMul_image(image_path)
        print (image.shape[0:2])
        images_grid, offsets, img_num = slide_window_det(image, (512, 512), overlap=0)
        for idx, image_grid in enumerate(tqdm(images_grid)):
            predictions = coco_demo.run_on_opencv_image_output_boxlist(image_grid)
            predictions.bbox += torch.Tensor(offsets[idx][::-1] + offsets[idx][::-1])
            if predictions.has_field("bquad"):
                predictions_bquad = predictions.get_field("bquad")
                predictions_bquad += torch.Tensor(offsets[idx][::-1] + 4*[0, 0])
                predictions.add_field("bquad", predictions_bquad)
            else:
                continue
            predictions_list.append(predictions)

            # image_grid_draw = image_grid.copy()
            # image = coco_demo.overlay_tqr_boxes(image_grid_draw, predictions)
            # image_path.replace("jpg_test", "output")
            # cv2.imwrite(image_path.replace("jpg_test", "output"), image)

            # break
            # embed()
            # predictions, features = coco_demo.run_on_opencv_image(image)
            # embed()
            # visFeature(features, out_dir)
        final_predictions = cat_boxlist(predictions_list)
        image_pan_draw = image[:, :, 4:5].copy()
        image_pan_draw = np.concatenate((image_pan_draw, image_pan_draw, image_pan_draw), axis=2)
        image_mul_draw = image[:, :, 0:3][:, :, [2, 1, 0]].copy()

        image_pan_draw = coco_demo.overlay_tqr_boxes(image_pan_draw, final_predictions)
        image_mul_draw = coco_demo.overlay_tqr_boxes(image_mul_draw, final_predictions)

        print (image_path.replace("tif_for_draw", "output_overlap_0"))

        cv2.imwrite(image_path.replace("tif_for_draw", "output_overlap_0").replace("_mul_upsample", "_pan"), image_pan_draw)
        cv2.imwrite(image_path.replace("tif_for_draw", "output_overlap_0"), image_mul_draw)


def gen_mul_upsample():
    images_path = '/data/lqp2018/jky/model/maskrcnn-benchmark-8008/demo_from_4002/output/shangdong_pansharpening/tif_for_draw'
    image_path_list = [os.path.join(images_path, x) for x in os.listdir(images_path) if is_image_file(x, ['_mul.tif'])]
    image_path_list.sort()
    print(image_path_list)
    for image_path in tqdm(image_path_list):
        # image = cv2.imread(image_path)
        predictions_list = []
        image = load_PanMul_image_save_mul_upsample(image_path)


def main_test():
    images_path = '/data/lqp2018/jky/model/maskrcnn-benchmark-8008/demo_from_4002/output/coco_qinghai/jpg_for_draw'
    out_dir = '/data/lqp2018/jky/model/maskrcnn-benchmark-8008/demo_from_4002/output/coco_qinghai/output'
    image_path_list = [os.path.join(images_path, x) for x in os.listdir(images_path) if is_image_file(x, ['.jpg'])]
    image_path_list.sort()
    predictions_list = []
    for image_path in tqdm(image_path_list):
        # image = cv2.imread(image_path)
        image = load_image(image_path)
        print (image.shape[0:2])
        image = coco_demo.run_on_opencv_image_no_features(image)
        image_path.replace("jpg_for_draw", "output")
        cv2.imwrite(image_path.replace("jpg_for_draw", "output_overlap_0"), image)


if __name__ == "__main__":
    main()
    # tif2jpg("/data/lqp2018/jky/model/maskrcnn-benchmark-8008/demo_from_4002/output/coco_qinghai/pic")




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

