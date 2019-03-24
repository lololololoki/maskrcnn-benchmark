# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_pbox_feature_extractors import make_roi_box_feature_extractor
from .roi_pbox_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .inference import make_roi_pbox_post_processor
from .loss import make_roi_box_loss_evaluator

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_pbox_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)

        _, bquad_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor(bquad_regression, proposals)
            return x, result, {}

        loss_bquad = self.loss_evaluator(bquad_regression=bquad_regression, proposals=proposals, targets=targets)

        return x, all_proposals, dict(loss_bquad=loss_bquad)

        # if self.training:
        #     # Faster R-CNN subsamples during training the proposals with a fixed
        #     # positive / negative ratio
        #     with torch.no_grad():
        #         proposals = self.loss_evaluator.subsample(proposals, targets)
        #
        # # extract features that will be fed to the final classifier. The
        # # feature_extractor generally corresponds to the pooler + heads
        # x = self.feature_extractor(features, proposals)
        # # final classifier that converts the features into predictions
        # class_logits, box_regression = self.predictor(x)
        #
        # if not self.training:
        #     result = self.post_processor((class_logits, box_regression), proposals)
        #     return x, result, {}
        #
        # loss_classifier, loss_box_reg = self.loss_evaluator(
        #     [class_logits], [box_regression]
        # )
        # return (
        #     x,
        #     proposals,
        #     dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        # )


def build_roi_pbox_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
