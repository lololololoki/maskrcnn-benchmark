# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from . import fpn as fpn_module
from . import resnet, MobileNetV2
from . import myATfpn as ATfpn_module


def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model
    
def build_resnet_atfpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    ATfpn = ATfpn_module.ATFPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("ATfpn", ATfpn)]))
    return model

def build_mobilenetv2_backbone(cfg):
    body = MobileNetV2.myMobileNetV2()
    model = nn.Sequential(OrderedDict([("body", body)]))
    return model


_BACKBONES = {"resnet": build_resnet_backbone, "resnet-fpn": build_resnet_fpn_backbone,
              "MobileNetV2": build_mobilenetv2_backbone}


def build_backbone(cfg):
    # assert cfg.MODEL.BACKBONE.CONV_BODY.startswith(
    #     "R-"
    # ), "Only ResNet and ResNeXt models are currently implemented"
    # # Models using FPN end with "-FPN"
    # if cfg.MODEL.BACKBONE.CONV_BODY.endswith("-FPN"):
    #     return build_resnet_fpn_backbone(cfg)
    # return build_resnet_backbone(cfg)

    ###jky###
    if cfg.MODEL.BACKBONE.CONV_BODY.endswith("V2"):
        func = _BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY]
        return func(cfg)
    # Models using FPN end with "-FPN"
    if cfg.MODEL.BACKBONE.CONV_BODY.endswith("-FPN"):
        if cfg.MODEL.BACKBONE.USE_ATTENTION_FPN:
            return build_resnet_atfpn_backbone(cfg)
        else:
            return build_resnet_fpn_backbone(cfg)
    return build_resnet_backbone(cfg)
