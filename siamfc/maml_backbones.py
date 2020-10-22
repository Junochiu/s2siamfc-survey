from __future__ import absolute_import

import ipdb
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from .googlenet import Inception3
from .vgg import VGG
from .maml_basicstructure import extract_top_level_dict, MetaBatchNormLayer, MetaConv2dLayer, MetaConvNormLayerReLU, \
    MetaLinearLayer

__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3']


class AlexNet(nn.Module):
    r"""
    AlexNet
    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self, args):
        super(AlexNet, self).__init__()
        self.conv1 = MetaConvNormLayerReLU(3, 96, stride=2, kernel_size=11, padding=0, args=args)
        self.pool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv2 = MetaConvNormLayerReLU(48, 256, 1, 5, 0, args=args,groups=2)
        self.pool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv3 = MetaConvNormLayerReLU(256, 384, 1, 3, 0, args=args)
        self.conv4 = MetaConvNormLayerReLU(192, 384, 1, 3, 0, args=args,groups=2)
        self.conv5 = MetaConvNormLayerReLU(192, 256, 1, 3, 0, relu=False, normalization=False, args=args,groups=2)
        # in original model, normalization should be true. However, the model loaded doesn'y come up with bn layer

    def forward(self, x, params, num_step):
        param = extract_top_level_dict(params)
        param_backbone = extract_top_level_dict(param['backbone'])
        x = self.conv1.forward(x, num_step, params=param_backbone['conv1'])
        x = self.pool1(x)
        x = self.conv2.forward(x, num_step, params=param_backbone['conv2'])
        x = self.pool2(x)
        x = self.conv3.forward(x, num_step, params=param_backbone['conv3'])
        x = self.conv4.forward(x, num_step, params=param_backbone['conv4'])
        x = self.conv5.forward(x, num_step, params=param_backbone['conv5'])
        return x


class Resnet18(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet18, self).__init__()
        self.backbone = resnet18(used_layers=[2, 3, 4])

    def forward(self, x):
        out = self.backbone(x)
        return out[-1]  # last conv


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.backbone = Inception3()
        self.backbone.update_params()

    def forward(self, x):
        out = self.backbone(x)
        return out


class VGG16(nn.Module):
    def __init__(self, ):
        super(VGG16, self).__init__()
        self.backbone = VGG()

    def forward(self, x):
        out = self.backbone(x)
        return out
