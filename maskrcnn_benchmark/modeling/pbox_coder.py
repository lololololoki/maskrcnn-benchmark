# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class PboxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights=(10., 10., 10., 10., 10., 10., 10., 10., 10., 10.), pbbox_xform_clip=math.log(1000. / 16), cfg=None):
        """
        Arguments:
            weights (10-element tuple)
            pbbox_xform_clip (float)
        """
        self.weights = weights
        self.pbbox_xform_clip = pbbox_xform_clip
        self.cfg = cfg

    def encode(self, reference_pboxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_pboxes (Tensor): reference pboxes (x; y;w1; h1;w2; h2;w3; h3;w4; h4)
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_w1 = reference_pboxes[:, 2]
        gt_h1 = reference_pboxes[:, 3]
        gt_w2 = reference_pboxes[:, 4]
        gt_h2 = reference_pboxes[:, 5]
        gt_w3 = reference_pboxes[:, 6]
        gt_h3 = reference_pboxes[:, 7]
        gt_w4 = reference_pboxes[:, 8]
        gt_h4 = reference_pboxes[:, 9]
        gt_ctr_x = reference_pboxes[:, 0]
        gt_ctr_y = reference_pboxes[:, 1]

        wx, wy, ww, wh, _, _ , _, _, _, _= self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw1 = ww * (gt_w1 / ex_widths)
        targets_dh1 = wh * (gt_h1 / ex_heights)
        targets_dw2 = ww * (gt_w2 / ex_widths)
        targets_dh2 = wh * (gt_h2 / ex_heights)
        targets_dw3 = ww * (gt_w3 / ex_widths)
        targets_dh3 = wh * (gt_h3 / ex_heights)
        targets_dw4 = ww * (gt_w4 / ex_widths)
        targets_dh4 = wh * (gt_h4 / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw1, targets_dh1, targets_dw2, targets_dh2, targets_dw3, targets_dh3, targets_dw4, targets_dh4), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative pbox offsets,
        get the decoded pboxes.

        Arguments:
            rel_codes (Tensor): encoded pboxes (x; y;w1; h1;w2; h2;w3; h3;w4; h4)
            boxes (Tensor): reference boxes.
        Return:
            pbox (Tensor): (x; y;w1; h1;w2; h2;w3; h3;w4; h4)
        """

        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh, _, _, _, _, _, _ = self.weights
        dx = rel_codes[:, 0::10] / wx
        dy = rel_codes[:, 1::10] / wy
        dw1 = rel_codes[:, 2::10] / ww
        dh1 = rel_codes[:, 3::10] / wh
        dw2 = rel_codes[:, 4::10] / ww
        dh2 = rel_codes[:, 5::10] / wh
        dw3 = rel_codes[:, 6::10] / ww
        dh3 = rel_codes[:, 7::10] / wh
        dw4 = rel_codes[:, 8::10] / ww
        dh4 = rel_codes[:, 9::10] / wh

        # Prevent sending too large values into torch.exp()
        # dw1 = torch.clamp(dw1, max=self.bbox_xform_clip)
        # dh1 = torch.clamp(dh1, max=self.bbox_xform_clip)
        # dw2 = torch.clamp(dw2, max=self.bbox_xform_clip)
        # dh2 = torch.clamp(dh2, max=self.bbox_xform_clip)
        # dw3 = torch.clamp(dw3, max=self.bbox_xform_clip)
        # dh3 = torch.clamp(dh3, max=self.bbox_xform_clip)
        # dw4 = torch.clamp(dw4, max=self.bbox_xform_clip)
        # dh4 = torch.clamp(dh4, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w1 = (dw1) * widths[:, None]
        pred_h1 = (dh1) * heights[:, None]
        pred_w2 = (dw2) * widths[:, None]
        pred_h2 = (dh2) * heights[:, None]
        pred_w3 = (dw3) * widths[:, None]
        pred_h3 = (dh3) * heights[:, None]
        pred_w4 = (dw4) * widths[:, None]
        pred_h4 = (dh4) * heights[:, None]

        pred_bquad = torch.zeros_like(rel_codes)
        pred_bquad[:, 0::10] = pred_ctr_x
        pred_bquad[:, 1::10] = pred_ctr_y
        pred_bquad[:, 2::10] = pred_w1
        pred_bquad[:, 3::10] = pred_h1
        pred_bquad[:, 4::10] = pred_w2
        pred_bquad[:, 5::10] = pred_h2
        pred_bquad[:, 6::10] = pred_w3
        pred_bquad[:, 7::10] = pred_h3
        pred_bquad[:, 8::10] = pred_w4
        pred_bquad[:, 9::10] = pred_h4

        return pred_bquad

    # def bquad2reformatted(self, bquad, reformatted_bquad):
    #     """
    #     From a set of original boxes and encoded relative pbox offsets,
    #     get the decoded pboxes.
    #
    #     Arguments:
    #         bquad (Tensor): (x1, y1, x2, y2, x3, y3, x4, y4)
    #         reformatted_bquad (Tensor): (x, y, w1, h1, w2, h2, w3, h3, w4, h4)
    #     Return:
    #         pbox (Tensor): (x; y;w1; h1;w2; h2;w3; h3;w4; h4)
    #     """
    #
    #     x1 = bquad[:,0]
    #     y1 = bquad[:,1]
    #     x2 = bquad[:,2]
    #     y2 = bquad[:,3]
    #     x3 = bquad[:,4]
    #     y3 = bquad[:,5]
    #     x4 = bquad[:,6]
    #     y4 = bquad[:,7]
    #     # boxes = boxes.to(rel_codes.dtype)
    #     #
    #     # TO_REMOVE = 1  # TODO remove
    #     # widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
    #     # heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
    #     # ctr_x = boxes[:, 0] + 0.5 * widths
    #     # ctr_y = boxes[:, 1] + 0.5 * heights
    #     #
    #     # wx, wy, ww, wh, _, _, _, _, _, _ = self.weights
    #     # dx = rel_codes[:, 0] / wx
    #     # dy = rel_codes[:, 1] / wy
    #     # dw1 = rel_codes[:, 2] / ww
    #     # dh1 = rel_codes[:, 3] / wh
    #     # dw2 = rel_codes[:, 4] / ww
    #     # dh2 = rel_codes[:, 5] / wh
    #     # dw3 = rel_codes[:, 6] / ww
    #     # dh3 = rel_codes[:, 7] / wh
    #     # dw4 = rel_codes[:, 8] / ww
    #     # dh4 = rel_codes[:, 9] / wh
    #     #
    #     # # Prevent sending too large values into torch.exp()
    #     # # dw1 = torch.clamp(dw1, max=self.bbox_xform_clip)
    #     # # dh1 = torch.clamp(dh1, max=self.bbox_xform_clip)
    #     # # dw2 = torch.clamp(dw2, max=self.bbox_xform_clip)
    #     # # dh2 = torch.clamp(dh2, max=self.bbox_xform_clip)
    #     # # dw3 = torch.clamp(dw3, max=self.bbox_xform_clip)
    #     # # dh3 = torch.clamp(dh3, max=self.bbox_xform_clip)
    #     # # dw4 = torch.clamp(dw4, max=self.bbox_xform_clip)
    #     # # dh4 = torch.clamp(dh4, max=self.bbox_xform_clip)
    #     #
    #     # pred_ctr_x = dx * widths + ctr_x
    #     # pred_ctr_y = dy * heights + ctr_y
    #     # pred_w1 = (dw1) * widths
    #     # pred_h1 = (dh1) * heights
    #     # pred_w2 = (dw2) * widths
    #     # pred_h2 = (dh2) * heights
    #     # pred_w3 = (dw3) * widths
    #     # pred_h3 = (dh3) * heights
    #     # pred_w4 = (dw4) * widths
    #     # pred_h4 = (dh4) * heights
    #     #
    #     # pred_boxes = torch.stack((pred_ctr_x, pred_ctr_y, pred_w1, pred_h1, pred_w2, pred_h2, pred_w3,
    #     #                        pred_h3, pred_w4, pred_h4), dim=1)
    #
    #     return

    def bquad2MBR(self, bquads):
        """
        From a set of original bquad ,
        get bquad's minimum bounding rectangle (MBR) bbox.

        Arguments:
            bquads (Tensor): bquad points (x, y, w1, h1, w2, h2, w3, h3, w4, h4)
        Return:
            every bquad's MBR Bbox (Tensor): (x, y, w, h)
        """

        TO_REMOVE = 1  # TODO remove

        wx, wy, ww, wh, _, _, _, _, _, _ = self.weights
        x = bquads[:, 0::10]
        y = bquads[:, 1::10]
        w1 = bquads[:, 2::10]
        h1 = bquads[:, 3::10]
        w2 = bquads[:, 4::10]
        h2 = bquads[:, 5::10]
        w3 = bquads[:, 6::10]
        h3 = bquads[:, 7::10]
        w4 = bquads[:, 8::10]
        h4 = bquads[:, 9::10]

        x1 = x + w1
        y1 = y + h1
        x2 = x + w2
        y2 = y + h2
        x3 = x + w3
        y3 = y + h3
        x4 = x + w4
        y4 = y + h4

        x_min, _ = torch.min(torch.cat([x1, x2, x3, x4], dim=1), dim=1)
        x_max, _ = torch.max(torch.cat([x1, x2, x3, x4], dim=1), dim=1)
        y_min, _ = torch.min(torch.cat([y1, y2, y3, y4], dim=1), dim=1)
        y_max, _ = torch.max(torch.cat([y1, y2, y3, y4], dim=1), dim=1)

        w = (x_max - x_min) / 10 + TO_REMOVE
        h = (y_max - y_min) / 10 + TO_REMOVE






        # Prevent sending too large values into torch.exp()
        # dw1 = torch.clamp(dw1, max=self.bbox_xform_clip)
        # dh1 = torch.clamp(dh1, max=self.bbox_xform_clip)
        # dw2 = torch.clamp(dw2, max=self.bbox_xform_clip)
        # dh2 = torch.clamp(dh2, max=self.bbox_xform_clip)
        # dw3 = torch.clamp(dw3, max=self.bbox_xform_clip)
        # dh3 = torch.clamp(dh3, max=self.bbox_xform_clip)
        # dw4 = torch.clamp(dw4, max=self.bbox_xform_clip)
        # dh4 = torch.clamp(dh4, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w1 = (dw1) * widths[:, None]
        pred_h1 = (dh1) * heights[:, None]
        pred_w2 = (dw2) * widths[:, None]
        pred_h2 = (dh2) * heights[:, None]
        pred_w3 = (dw3) * widths[:, None]
        pred_h3 = (dh3) * heights[:, None]
        pred_w4 = (dw4) * widths[:, None]
        pred_h4 = (dh4) * heights[:, None]

        pred_bquad = torch.zeros_like(rel_codes)
        pred_bquad[:, 0::10] = pred_ctr_x
        pred_bquad[:, 1::10] = pred_ctr_y
        pred_bquad[:, 2::10] = pred_w1
        pred_bquad[:, 3::10] = pred_h1
        pred_bquad[:, 4::10] = pred_w2
        pred_bquad[:, 5::10] = pred_h2
        pred_bquad[:, 6::10] = pred_w3
        pred_bquad[:, 7::10] = pred_h3
        pred_bquad[:, 8::10] = pred_w4
        pred_bquad[:, 9::10] = pred_h4

        return pred_bquad
