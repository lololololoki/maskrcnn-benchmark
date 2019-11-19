# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from maskrcnn_benchmark.modeling.backbone.resnet import _make_stage, BottleneckWithFixedBatchNormDoubleHead
from maskrcnn_benchmark.layers import Conv2d
import math
from torch.nn import functional as F

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


class FPNPredictorDoubleHead(nn.Module):
    def __init__(self, cfg):
        super(FPNPredictorDoubleHead, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        # representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2

        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6_tqr_fc_head = nn.Linear(input_size, representation_size)
        self.fc7_tqr_fc_head = nn.Linear(representation_size, representation_size)

        for l in [self.fc6_tqr_fc_head, self.fc7_tqr_fc_head]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        self.cls_score_cls_tqr = nn.Linear(representation_size, num_classes)
        self.pred_cls_tqr = nn.Linear(representation_size, num_classes * 10)

        # representation_size = cfg.MODEL.TQR.ROI_TQR_HEAD.CONV_HEAD_CHANNELS

        self.upchannels_conv1x1_tqr = Conv2d(
            cfg.MODEL.BACKBONE.OUT_CHANNELS,
            representation_size,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.upchannels_right_convs_tqr = nn.Sequential(
                # 3x3x256 conv
                nn.Conv2d(cfg.MODEL.BACKBONE.OUT_CHANNELS, cfg.MODEL.BACKBONE.OUT_CHANNELS, 3, 1, 1, bias=False),
                nn.BatchNorm2d(cfg.MODEL.BACKBONE.OUT_CHANNELS),
                # 1x1x1024 cnv
                nn.Conv2d(cfg.MODEL.BACKBONE.OUT_CHANNELS, representation_size, 1, 1, 0, bias=False),
                nn.BatchNorm2d(representation_size),
            )

        self.convs_reg_tqr = _make_stage(
            BottleneckWithFixedBatchNormDoubleHead,
            representation_size,
            256,
            representation_size,
            cfg.MODEL.TQR.ROI_TQR_HEAD.REG_CONVS_NUM,
            1,
            True,
            1,
        )

        self.reg_avgpool_tqr = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc_reg_tqr = nn.Linear(representation_size, representation_size)

        self.cls_score_reg_tqr = nn.Linear(representation_size, num_classes)
        self.pred_reg_tqr = nn.Linear(representation_size, num_classes * 10)

        self._initialize_weights()

        # other init
        nn.init.normal_(self.cls_score_cls_tqr.weight, std=0.01)
        nn.init.normal_(self.pred_cls_tqr.weight, std=0.001)
        for l in [self.cls_score_cls_tqr, self.pred_cls_tqr]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x_cls = x
        x_cls = x_cls.view(x_cls.size(0), -1)
        x_cls = F.relu(self.fc6_tqr_fc_head(x_cls))
        x_cls = F.relu(self.fc7_tqr_fc_head(x_cls))

        scores_cls = self.cls_score_cls_tqr(x_cls)
        pred_cls = self.pred_cls_tqr(x_cls)

        # residual = x

        x_l = self.upchannels_conv1x1_tqr(x)
        x_r = self.upchannels_right_convs_tqr(x)
        x = x_l + x_r
        x = F.relu(x)

        x = self.convs_reg_tqr(x)

        x = self.reg_avgpool_tqr(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc_reg_tqr(x))

        scores_reg = self.cls_score_reg_tqr(x)
        pred_reg = self.pred_reg_tqr(x)

        return scores_cls, pred_reg

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobileNetV2Predictor(nn.Module):
    def __init__(self, config, pretrained=None):
        super(MobileNetV2Predictor, self).__init__()

        # stage_index = 4
        # stage2_relative_factor = 2 ** (stage_index - 1)
        # res2_out_channels = config.MODEL.RESNETS.RES2_OUT_CHANNELS
        # num_inputs = res2_out_channels * stage2_relative_factor

        out_channels = 1280

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.cls_score = nn.Linear(out_channels, num_classes)
        self.bbox_pred = nn.Linear(out_channels, num_classes * 10)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred
        
     
class FPNMobileQuadPredictor(nn.Module):
    def __init__(self, cfg):
        super(FPNMobileQuadPredictor, self).__init__()

        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = int((cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM / 4) * 10)
        self.conv_linear4quad = nn.Conv2d(256, representation_size, 7, 1, 0)
        self.cls_score_pbox = nn.Linear(representation_size, num_classes)
        self.pred_pbox = nn.Linear(representation_size, num_classes * 10)

        nn.init.normal_(self.cls_score_pbox.weight, std=0.01)
        nn.init.normal_(self.pred_pbox.weight, std=0.001)
        for l in [self.cls_score_pbox, self.pred_pbox]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = self.conv_linear4quad(x)
        x = x.view(x.size(0), -1)
        scores = self.cls_score_pbox(x)
        pred_pbox = self.pred_pbox(x)

        return scores, pred_pbox


_ROI_BOX_PREDICTOR = {
    "FastRCNNPredictor": FastRCNNPredictor,
    "FPNPredictor": FPNPredictor,
    "MobileNetV2Predictor": MobileNetV2Predictor,
    "FPNMobileQuadPredictor": FPNMobileQuadPredictor,
    "FPNPredictorDoubleHead": FPNPredictorDoubleHead
}


def make_roi_box_predictor(cfg):
    func = _ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg)
