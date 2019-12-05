# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from torchvision.transforms import functional as F

class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # add pan & mul
        img_pan = Image.open(os.path.join(self.root.replace('images', 'pan'),
                                          path.replace('.jpg', '_pan.tif')))
        img_pan = np.expand_dims(np.array(img_pan), 2)
        img_mul_upsample = Image.open(os.path.join(self.root.replace('images', 'mul_upsample'),
                                                   path.replace('.jpg', '_mul.tif')))

        img_pan_mul = Image.open(os.path.join(self.root.replace('images', 'pan_mul'),
                                                   path.replace('.jpg', '.tif')))

        if self.transform is not None:
            img_pan = self.transform(img)
            img_mul_upsample = self.transform(img_mul_upsample)
            img_pan_mul = self.transform(img_pan_mul)

        return img, img_pan, img_mul_upsample, img_pan_mul, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class COCODataset(CocoDetection):
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
        img, img_pan, img_mul_upsample, img_pan_mul, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes
        if not boxes:
            raise ValueError("Image id {} ({}) doesn't have boxes annotations!".format(self.ids[idx], anno))

        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]

        # guard against no masks
        if not masks:
            raise ValueError("Image id {} ({}) doesn't have masks annotations!".format(self.ids[idx], anno))

        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        # TODO bquads can not be resized now
        if "bquad" in anno[0]:
            bquads = [obj["bquad"] for obj in anno]
            new_bquads = []
            HEIGHT = 512
            WIDTH = 512
            ratio_w = float(WIDTH) / float(img.size[0])
            ratio_h = float(HEIGHT) / float(img.size[1])
            if ratio_w != 1 or ratio_h != 1:
                for bquad in bquads:
                    bquad = [x*ratio_w for x in bquad]
                    new_bquads.append(bquad)
            else:
                new_bquads = bquads
            bquads = torch.as_tensor(new_bquads).reshape(-1, 10)
            target.add_field("bquads", bquads)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            # img_pan, target_temp = self.transforms(img_pan, target)

            def tramsforms_img(img, if_mul = False):
                img = F.to_tensor(img)
                # if if_mul:
                #     img = img[[2, 1, 0, 3]] * 255
                # else:
                #     img = img * 255
                # img = F.normalize(img, mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
                return img

            img_pan = tramsforms_img(img_pan)
            img_mul_upsample = tramsforms_img(img_mul_upsample)
            img_pan_mul = tramsforms_img(img_pan_mul)
            img, target = self.transforms(img, target)

            img_mul_pan = torch.cat((img_mul_upsample, img_pan, img_pan_mul), dim=0)

        return img_mul_pan, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
