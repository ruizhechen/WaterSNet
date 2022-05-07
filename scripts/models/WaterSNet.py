import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.Res2Net_v1b import res2net50_v1b_26w_4s

def cus_sample(feat, **kwargs):
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)

def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y

class Simam(nn.Module):
    def __init__(self, e=1e-4):
        super(Simam, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.e = e

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h -1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e)) + 0.5
        return x * self.sigmoid(y)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class RFB_m(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_m, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class AFB(nn.Module):
    def __init__(self, scale_factor1,scale_factor2,channel=64):
        super(AFB, self).__init__()
        self.scale_factor1=scale_factor1
        self.scale_factor2=scale_factor2
        self.simam=Simam()
        self.upsample = cus_sample
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x, y,z):
        y = self.upsample(y, scale_factor=self.scale_factor1)
        z = self.upsample(z, scale_factor=self.scale_factor2)
        xyz = x + y + z
        wei = self.simam(xyz)
        xo = x * wei + y * ((1-wei)/2) + z * ((1-wei)/2)
        xo = self.conv(xo)
        return xo

class Decoder1(nn.Module):
    def __init__(self,channel=64):
        super(Decoder1, self).__init__()
        self.upsample=cus_sample
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x):
        x=self.upsample(x,scale_factor=4)
        x=self.conv(x)
        return x

class Decoder2(nn.Module):
    def __init__(self,in_channel=128,out_channel=64):
        super(Decoder2, self).__init__()
        self.upsample=cus_sample
        self.conv1 = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.conv2 = BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x):
        x=self.conv1(x)
        x=self.upsample(x,scale_factor=2)
        x=self.conv2(x)
        return x

class WaterSNet(nn.Module):
    def __init__(self, channel=64):
        super(WaterSNet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.rfb1 = RFB_m(64, channel)
        self.rfb2 = RFB_m(256, channel)
        self.rfb3 = RFB_m(512, channel)
        self.rfb4 = RFB_m(1024, channel)
        self.rfb5 = RFB_m(2048, channel)
        self.afb123 = AFB(scale_factor1=1,scale_factor2=2)
        self.afb345 = AFB(scale_factor1=2,scale_factor2=4)       
        self.decoder1=Decoder1()
        self.decoder2=Decoder2()
        self.upconv123 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.upconv345 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.classifier = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1_rfb = self.rfb1(x)
        x = self.resnet.layer1(x)
        x2_rfb = self.rfb2(x)
        x = self.resnet.layer2(x)
        x3_rfb = self.rfb3(x)
        x = self.resnet.layer3(x)
        x4_rfb = self.rfb4(x)
        x = self.resnet.layer4(x)
        x5_rfb = self.rfb5(x)
        x123 = self.afb123(x1_rfb,x2_rfb,x3_rfb)
        x345 = self.afb345(x3_rfb,x4_rfb,x5_rfb)
        d32=torch.cat((self.upconv345(x345), self.decoder1(x5_rfb)), 1)
        d21=torch.cat((self.upconv123(x123), self.decoder2(d32)), 1)
        output=self.classifier(d21)
        output=F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        return F.sigmoid(output)


