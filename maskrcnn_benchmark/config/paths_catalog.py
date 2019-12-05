# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"

    DATASETS = {
        "coco_2014_train": (
            "coco/train2014",
            "coco/annotations/instances_train2014.json",
        ),
        "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
        "coco_2014_minival": (
            "coco/val2014",
            "coco/annotations/instances_minival2014.json",
        ),
        "coco_2014_valminusminival": (
            "coco/val2014",
            "coco/annotations/instances_valminusminival2014.json",
        ),
        "coco_mapping_challenge_train": (
            "coco_mapping_challenge/train/images",
            "coco_mapping_challenge/train/voc_0712_trainval_trec.json",
        ),
        "coco_mapping_challenge_val": (
            "coco_mapping_challenge/val/images",
            "coco_mapping_challenge/val/voc_0712_trainval_trec.json",
        ),
        "coco_mapping_challenge_train_small": (
            "coco_mapping_challenge/train/images",
            "coco_mapping_challenge/train/voc_0712_trainval_small_trec.json",
        ),
        "coco_mapping_challenge_val_small": (
            "coco_mapping_challenge/val/images",
            "coco_mapping_challenge/val/voc_0712_trainval_small_trec.json",
        ),
        "coco_qinghai_train": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_trainval.json",
        ),
        "coco_qinghai_test": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_test.json",
        ),
        "coco_qinghai_train_bquad": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_trainval_bquad.json",
        ),
        "coco_qinghai_test_bquad": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_test_bquad.json",
        ),
        "coco_qinghai_train_trec": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_trainval_trec.json",
        ),
        "coco_qinghai_test_trec": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_test_trec.json",
        ),
        "coco_qinghai_train_bquad_greater_than_4_rec": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_trainval_bquad_greater_than_4_rec.json",
        ),
        "coco_qinghai_test_bquad_greater_than_4_rec": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_test_bquad_greater_than_4_rec.json",
        ),
        "coco_qinghai_train_bquad_only_4points_minarea5": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_trainval_bquad_only_4points_minarea5.json",
        ),
        "coco_qinghai_test_bquad_only_4points_minarea5": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_test_bquad_only_4points_minarea5.json",
        ),
        "coco_qinghai_train_bquad_transform_into_4points_minarea5_sort": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_trainval_bquad_transform_into_4points_minarea5_sort.json",
        ),
        "coco_qinghai_test_bquad_transform_into_4points_minarea5_sort": (
            "coco_qinghai/images",
            "coco_qinghai/voc_0712_test_bquad_transform_into_4points_minarea5_sort.json",
        ),
        "coco_qinghai_urban_train_trec": (
            "coco_qinghai/VOC0712_urban/images",
            "coco_qinghai/VOC0712_urban/voc_0712_trainval_trec.json",
        ),
        "coco_qinghai_urban_test_trec": (
            "coco_qinghai/VOC0712_urban/images",
            "coco_qinghai/VOC0712_urban/voc_0712_test_trec.json",
        ),
        "coco_qinghai_suburban_train_trec": (
            "coco_qinghai/VOC0712_suburban/images",
            "coco_qinghai/VOC0712_suburban/voc_0712_trainval_trec.json",
        ),
        "coco_qinghai_suburban_test_trec": (
            "coco_qinghai/VOC0712_suburban/images",
            "coco_qinghai/VOC0712_suburban/voc_0712_test_trec.json",
        ),
        "coco_qinghai_suburban_correct_train_trec": (
            "coco_qinghai/VOC0712_suburban_correct/images",
            "coco_qinghai/VOC0712_suburban_correct/voc_0712_trainval_trec.json",
        ),
        "coco_qinghai_suburban_correct_test_trec": (
            "coco_qinghai/VOC0712_suburban_correct/images",
            "coco_qinghai/VOC0712_suburban_correct/voc_0712_test_trec.json",
        ),
        "coco_qinghai_rural_train_trec": (
            "coco_qinghai/VOC0712_rural/images",
            "coco_qinghai/VOC0712_rural/voc_0712_trainval_trec.json",
        ),
        "coco_qinghai_rural_test_trec": (
            "coco_qinghai/VOC0712_rural/images",
            "coco_qinghai/VOC0712_rural/voc_0712_test_trec.json",
        ),
        "coco_fujian_train": (
            "coco_fujian/images",
            "coco_fujian/voc_0712_trainval.json",
        ),
        "coco_fujian_test": (
            "coco_fujian/images",
            "coco_fujian/voc_0712_test.json",
        ),
        "coco_fujian_train_trec": (
            "coco_fujian/images",
            "coco_fujian/voc_0712_trainval_trec.json",
        ),
        "coco_fujian_test_trec": (
            "coco_fujian/images",
            "coco_fujian/voc_0712_test_trec.json",
        ),
        "coco_fujian_train_bquad": (
            "coco_fujian/images",
            "coco_fujian/voc_0712_trainval_bquad.json",
        ),
        "coco_fujian_test_bquad": (
            "coco_fujian/images",
            "coco_fujian/voc_0712_test_bquad.json",
        ),
        "coco_canada_demo": (
            "coco_canada_demo/images",
            "coco_canada_demo/voc_0712_test_trec.json",
        ),
        "coco_canada_1_demo": (
            "coco_canada_1_demo/images",
            "coco_canada_1_demo/voc_0712_test_trec.json",
        ),
        "coco_canada_2_demo": (
            "coco_canada_2_demo/images",
            "coco_canada_2_demo/voc_0712_test_trec.json",
        ),
        "coco_shandong_pansharpening_test_trec": (
            "coco_shandong_pansharpening/images",
            "coco_shandong_pansharpening/voc_0712_test_trec.json",
        ),
        "coco_shandong_pansharpening_train_trec": (
            "coco_shandong_pansharpening/images",
            "coco_shandong_pansharpening/voc_0712_trainval_trec.json",
        ),
    }

    @staticmethod
    def get(name):
        if "pansharpening" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="COCODatasetPansharpening",
                args=args,
            )
        elif "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://s3-us-west-2.amazonaws.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
