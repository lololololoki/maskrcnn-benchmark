# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn


class FastRCNNPredictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(FastRCNNPredictor, self).__init__()

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        num_inputs = res2_out_channels * stage2_relative_factor

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score_pbbox = nn.Linear(num_inputs, num_classes)
        self.pred_pbbox = nn.Linear(num_inputs, num_classes * 10)

        nn.init.normal_(self.cls_score_pbbox.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score_pbbox.bias, 0)

        nn.init.normal_(self.pred_pbbox.weight, mean=0, std=0.001)
        nn.init.constant_(self.pred_pbbox.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_score_pbbox = self.cls_score_pbbox(x)
        pred_pbbox = self.pred_pbbox(x)
        return cls_score_pbbox, pred_pbbox


class FPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        self.cls_score_pbox = nn.Linear(representation_size, num_classes)
        self.pred_pbox = nn.Linear(representation_size, num_classes * 10)

        nn.init.normal_(self.cls_score_pbox.weight, std=0.01)
        nn.init.normal_(self.pred_pbox.weight, std=0.001)
        for l in [self.cls_score_pbox, self.pred_pbox]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.cls_score_pbox(x)
        pred_pbox = self.pred_pbox(x)

        return scores, pred_pbox


_ROI_BOX_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "FPNPredictor": FPNPredictor,
}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
