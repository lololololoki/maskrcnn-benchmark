# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
            bquads = torch.as_tensor(bquads).reshape(-1, 10)
            target.add_field("bquads", bquads)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
