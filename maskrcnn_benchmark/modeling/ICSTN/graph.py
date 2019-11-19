import numpy as np
import torch
import time
from maskrcnn_benchmark.modeling.ICSTN import data,warp,util,options
from maskrcnn_benchmark.modeling.backbone import resnet

import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d

# build classification network
class FullCNN(torch.nn.Module):
	def __init__(self,opt):
		super(FullCNN,self).__init__()
		self.inDim = 1
		def conv2Layer(outDim):
			conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[3,3],stride=1,padding=0)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			self.inDim = outDim
			return fc
		def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(3),torch.nn.ReLU(True),
			conv2Layer(6),torch.nn.ReLU(True),maxpoolLayer(),
			conv2Layer(9),torch.nn.ReLU(True),
			conv2Layer(12),torch.nn.ReLU(True)
		)
		self.inDim *= 8**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(48),torch.nn.ReLU(True),
			linearLayer(opt.labelN)
		)
		initialize(opt,self,opt.stdC)
	def forward(self,opt,image):
		feat = image
		feat = self.conv2Layers(feat).view(opt.batchSize,-1)
		feat = self.linearLayers(feat)
		output = feat
		return output

# build classification network
class CNN(torch.nn.Module):
	def __init__(self,opt):
		super(CNN,self).__init__()
		self.inDim = 1
		def conv2Layer(outDim):
			conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[9,9],stride=1,padding=0)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			self.inDim = outDim
			return fc
		def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(3),torch.nn.ReLU(True)
		)
		self.inDim *= 20**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(opt.labelN)
		)
		initialize(opt,self,opt.stdC)
	def forward(self,opt,image):
		feat = image
		feat = self.conv2Layers(feat).view(opt.batchSize,-1)
		feat = self.linearLayers(feat)
		output = feat
		return output

# an identity class to skip geometric predictors
class Identity(torch.nn.Module):
	def __init__(self): super(Identity,self).__init__()
	def forward(self,opt,feat): return [feat]

# build Spatial Transformer Network
class STN(torch.nn.Module):
	def __init__(self,opt):
		super(STN,self).__init__()
		self.inDim = 1
		def conv2Layer(outDim):
			conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[7,7],stride=1,padding=0)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			self.inDim = outDim
			return fc
		def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(4),torch.nn.ReLU(True),
			conv2Layer(8),torch.nn.ReLU(True),maxpoolLayer()
		)
		self.inDim *= 8**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(48),torch.nn.ReLU(True),
			linearLayer(opt.warpDim)
		)
		initialize(opt,self,opt.stdGP,last0=True)
	def forward(self,opt,image):
		imageWarpAll = [image]
		feat = image
		feat = self.conv2Layers(feat).view(opt.batchSize,-1)
		feat = self.linearLayers(feat)
		p = feat
		pMtrx = warp.vec2mtrx(opt,p)
		imageWarp = warp.transformImage(opt,image,pMtrx)
		imageWarpAll.append(imageWarp)
		return imageWarpAll

class ICSTNBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride=1,
    ):
        super(ICSTNBottleneck, self).__init__()

        self.blocks = []
        self.downsample = None
        if in_channels != out_channels:

            self.downsample = True
            layer_name = "ICSTN_downsample_"
            module = nn.Sequential(
                Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                FrozenBatchNorm2d(out_channels),
            )
            nn.init.kaiming_normal_(module[0].weight, mode="fan_out", nonlinearity="relu")
            self.add_module(layer_name, module)
            self.blocks.append(layer_name)


        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (1, 1)

        layer_name = "ICSTN_conv1_"
        module = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        self.add_module(layer_name, module)
        self.blocks.append(layer_name)

        # self.ICSTN_bn1 = FrozenBatchNorm2d(bottleneck_channels)
        layer_name = "ICSTN_bn1_"
        module = FrozenBatchNorm2d(bottleneck_channels)
        self.add_module(layer_name, module)
        self.blocks.append(layer_name)

        # TODO: specify init for the above

        # self.ICSTN_conv2 = Conv2d(
        #     bottleneck_channels,
        #     bottleneck_channels,
        #     kernel_size=3,
        #     stride=stride_3x3,
        #     padding=1,
        #     bias=False,
        #     groups=num_groups,
        # )
        # self.ICSTN_bn2 = FrozenBatchNorm2d(bottleneck_channels)
        #
        # self.ICSTN_conv3 = Conv2d(
        #     bottleneck_channels, out_channels, kernel_size=1, bias=False
        # )
        # self.ICSTN_bn3 = FrozenBatchNorm2d(out_channels)

        layer_name = "ICSTN_conv2_"
        module = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1,
            bias=False,
            groups=num_groups,
        )
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        self.add_module(layer_name, module)
        self.blocks.append(layer_name)

        layer_name = "ICSTN_bn2_"
        module = FrozenBatchNorm2d(bottleneck_channels)
        self.add_module(layer_name, module)
        self.blocks.append(layer_name)

        layer_name = "ICSTN_conv3_"
        module = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        self.add_module(layer_name, module)
        self.blocks.append(layer_name)

        layer_name = "ICSTN_bn3_"
        module = FrozenBatchNorm2d(out_channels)
        self.add_module(layer_name, module)
        self.blocks.append(layer_name)

    def forward(self, x):
        residual = x

        # out = self.ICSTN_conv1(x)
        # out = self.ICSTN_bn1(out)
        # out = F.relu_(out)
        #
        # out = self.ICSTN_conv2(out)
        # out = self.ICSTN_bn2(out)
        # out = F.relu_(out)
        #
        # out0 = self.ICSTN_conv3(out)
        # out = self.ICSTN_bn3(out0)
        #
        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out = getattr(self, "ICSTN_conv1_")(x)
        out = getattr(self, "ICSTN_bn1_")(out)
        out = F.relu_(out)

        out = getattr(self, "ICSTN_conv2_")(out)
        out = getattr(self, "ICSTN_bn2_")(out)
        out = F.relu_(out)

        out = getattr(self, "ICSTN_conv3_")(out)
        out = getattr(self, "ICSTN_bn3_")(out)

        if self.downsample is not None:
            residual = getattr(self, "ICSTN_downsample_")(x)

        out += residual
        out = F.relu_(out)

        return out

# build Inverse Compositional STN
class ICSTN(torch.nn.Module):
	def __init__(self,opt,cfg):
		super(ICSTN,self).__init__()

		self.opt = opt
		self.pInit = data.genPerturbations(opt)

		self.cfg = cfg.clone()
		self.inDim = 256
		def conv2Layer(outDim):
			# conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[7,7],stride=1,padding=0)
			conv = ICSTNBottleneck(self.inDim, self.inDim // 2, outDim)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			# fc.weight.data.zero_()
			# fc.bias.data.zero_()
			# nn.init.kaiming_normal_(fc.weight, mode="fan_out", nonlinearity="relu")
			# nn.init.constant_(fc.bias, 0)
			self.inDim = outDim
			return fc
		# def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(512)
		)
		self.inDim *= cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(1024),torch.nn.ReLU(True),
			linearLayer(opt.warpDim)
		)
		# initialize(opt,self,opt.stdGP,last0=True)
	def forward(self,image):
		opt = self.opt
		batchSize = image.size(0)
		opt.batchSize = batchSize
		p = self.pInit.expand(batchSize,self.pInit.size(-1))
		imageWarpAll = []
		for l in range(opt.warpN):
			pMtrx = warp.vec2mtrx(opt,p)
			imageWarp = warp.transformImage(opt,image,pMtrx)
			imageWarpAll.append(imageWarp)
			feat = imageWarp
			feat = self.conv2Layers(feat).view(batchSize,-1)
			feat = self.linearLayers(feat)
			dp = feat
			p = warp.compose(opt,p,dp)
		pMtrx = warp.vec2mtrx(opt,p)
		imageWarp = warp.transformImage(opt,image,pMtrx)
		imageWarpAll.append(imageWarp)
		return imageWarp

# initialize weights/biases
def initialize(opt,model,stddev,last0=False):
	for m in model.conv2Layers:
		if isinstance(m,torch.nn.Conv2d):
			m.weight.data.normal_(0,stddev)
			m.bias.data.normal_(0,stddev)
	for m in model.linearLayers:
		if isinstance(m,torch.nn.Linear):
			if last0 and m is model.linearLayers[-1]:
				m.weight.data.zero_()
				m.bias.data.zero_()
			else:
				m.weight.data.normal_(0,stddev)
				m.bias.data.normal_(0,stddev)

# build Inverse Compositional STN
class ICSTN_mnist(torch.nn.Module):
	def __init__(self,opt):
		super(ICSTN,self).__init__()
		self.inDim = 1
		def conv2Layer(outDim):
			conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[7,7],stride=1,padding=0)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			self.inDim = outDim
			return fc
		def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(4),torch.nn.ReLU(True),
			conv2Layer(8),torch.nn.ReLU(True),maxpoolLayer()
		)
		self.inDim *= 8**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(48),torch.nn.ReLU(True),
			linearLayer(opt.warpDim)
		)
		initialize(opt,self,opt.stdGP,last0=True)
	def forward(self,opt,image,p):
		imageWarpAll = []
		for l in range(opt.warpN):
			pMtrx = warp.vec2mtrx(opt,p)
			imageWarp = warp.transformImage(opt,image,pMtrx)
			imageWarpAll.append(imageWarp)
			feat = imageWarp
			feat = self.conv2Layers(feat).view(opt.batchSize,-1)
			feat = self.linearLayers(feat)
			dp = feat
			p = warp.compose(opt,p,dp)
		pMtrx = warp.vec2mtrx(opt,p)
		imageWarp = warp.transformImage(opt,image,pMtrx)
		imageWarpAll.append(imageWarp)
		return imageWarpAll


# initialize weights/biases
def initialize(opt,model,stddev,last0=False):
	for m in model.conv2Layers:
		if isinstance(m,torch.nn.Conv2d):
			m.weight.data.normal_(0,stddev)
			m.bias.data.normal_(0,stddev)
	for m in model.linearLayers:
		if isinstance(m,torch.nn.Linear):
			if last0 and m is model.linearLayers[-1]:
				m.weight.data.zero_()
				m.bias.data.zero_()
			else:
				m.weight.data.normal_(0,stddev)
				m.bias.data.normal_(0,stddev)
