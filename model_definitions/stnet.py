import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from resnet_2d import Bottleneck, conv1x1, conv3x3
from temporal_xception import TemporalXception
from senet import SEResNeXtBottleneck
from collections import OrderedDict
import pretrainedmodels

import torchvision


class StNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            groups,
            reduction,
            dropout_p=0.2,
            inplanes=128,
            input_3x3=True,
            downsample_kernel_size=3,
            downsample_padding=1,
            num_classes=1000,
            T=7,
            N=5,
            input_channels=3,
    ):
        super(StNet, self).__init__()
        self.inplanes = inplanes
        self.T = T
        self.N = N
        layer0_modules = [
            (
                "conv1",
                nn.Conv2d(
                    3 * self.N, inplanes, kernel_size=7, stride=2, padding=3, bias=False
                ),
            ),
            ("bn1", nn.BatchNorm2d(inplanes)),
            ("relu1", nn.ReLU(inplace=True)),
        ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(("pool", nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )

        self.temp1 = TemporalBlock(512)

        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.temp2 = TemporalBlock(1024)
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )

        self.xception = TemporalXception(2048, 2048)
        self.last_linear = nn.Linear(2048, num_classes)

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            groups,
            reduction,
            stride=1,
            downsample_kernel_size=1,
            downsample_padding=0,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=downsample_kernel_size,
                    stride=stride,
                    padding=downsample_padding,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, groups, reduction, stride, downsample)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        # size (batch_size, channels, video_length = T * N, height, width)
        if len(x.shape) == 4:
            # Expand video channel
            x = x.unsqueeze(2)
        B, C, L, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        assert self.T * self.N == L
        x = x.view(B * self.T, self.N * C, H, W)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # size (batch_size*T, Ci, Hi, Wi)
        size = x.size()
        x = x.view(B, self.T, x.size(1), x.size(2), x.size(3))
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = self.temp1(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        x = self.layer3(x)
        # size (batch_size*T, Ci, Hi, Wi)
        size = x.size()
        x = x.view(B, self.T, x.size(1), x.size(2), x.size(3))
        B, T, C, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = self.temp2(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        x = self.layer4(x)
        # size (batch_size*T, Ci, Hi, Wi)
        size = x.size()
        x = F.avg_pool2d(x, kernel_size=(size[2], size[3]))
        # size (batch_size*T, Ci, 1, 1)
        x = x.view(B, self.T, size[1]).permute(0, 2, 1)
        # size (batch_size, T, Ci)
        x = self.xception(x)
        x = self.last_linear(x)

        return x


class TemporalBlock(nn.Module):
    def __init__(self, channels):
        super(TemporalBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            stride=1,
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.dirac_(m.weight)
                # m.weight.data.fill_(1 / (3 * self.channels))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


def load_weights(model, state):
    pretrained_dict = {}
    model_state = model.state_dict()
    for name, param in state.items():
        if name.startswith("layer0.conv1"):
            pretrained_dict[name] = state[name].repeat(1, model.N, 1, 1) / model.N
        else:
            pretrained_dict[name] = state[name]

    model_state.update(pretrained_dict)
    model.load_state_dict(model_state)
    return model


def stnet50(input_channels=3, num_classes=200, T=1, N=1, pretrained=False):
    """
    Construct stnet with a SE-Resnext 50 backbone.
    """

    model = StNet(
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        input_channels=input_channels,
        num_classes=num_classes,
        T=T,
        N=N
    )
    if pretrained:
        model = load_weights(
            model,
            pretrainedmodels.__dict__["se_resnext50_32x4d"](
                num_classes=num_classes, pretrained="imagenet"
            ).state_dict(),
        )


    return model
