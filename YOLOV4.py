import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=True, act='leaky'):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False if bn else True)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=1E-4) if bn else nn.Identity()
        self.activation = nn.LeakyReLU(0.1, inplace=True) if act == 'leaky' else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, return_layer=False):
        super(BasicBlock, self).__init__()
        self.return_layer = return_layer
        self.conv1 = BasicConv(in_channels, in_channels, kernel_size=3, stride=1, bn=True, act='leaky')
        self.conv2 = BasicConv(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, bn=True, act='leaky')
        self.conv3 = BasicConv(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, bn=True, act='leaky')
        self.conv4 = BasicConv(in_channels, in_channels, kernel_size=1, stride=1, bn=True, act='leaky')
    def forward(self, x):
        x1 = self.conv1(x)
        x1_route = x1[:, x1.shape[1]//2:, :, :]
        x2 = self.conv2(x1_route)
        x3 = self.conv3(x2)
        x2_x3_route = torch.cat([x3, x2], dim=1)
        x4 = self.conv4(x2_x3_route)
        x = torch.cat([x1, x4], dim=1)
        return [x, x4] if self.return_layer else [x]


class YOLOV4(nn.Module):
    def __init__(self, B=3, C=80):
        super(YOLOV4, self).__init__()
        in_channels = 3
        yolo_channels = (5 + C) * B

        self.conv1 = BasicConv(in_channels, 32, kernel_size=3, stride=2, bn=True, act='leaky')
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2, bn=True, act='leaky')
        self.block1 = BasicBlock(64, 128, return_layer=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = BasicBlock(128, 256, return_layer=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = BasicBlock(256, 512, return_layer=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = BasicConv(512, 512, kernel_size=3, stride=1, bn=True, act='leaky')
        self.conv4 = BasicConv(512, 256, kernel_size=1, stride=1, bn=True, act='leaky')

        self.conv5 = BasicConv(256, 512, kernel_size=3, stride=1, bn=True, act='leaky')
        self.conv6 = BasicConv(512, yolo_channels, kernel_size=1, stride=1, bn=False, act='linear')

        self.conv7 = BasicConv(256, 128, kernel_size=1, stride=1, bn=True, act='leaky')
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = BasicConv(384, 256, kernel_size=3, stride=1, bn=True, act='leaky')
        self.conv9 = BasicConv(256, yolo_channels, kernel_size=1, stride=1, bn=False, act='linear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        block1 = self.block1(x)
        x = self.pool1(block1[0])
        block2 = self.block2(x)
        x = self.pool2(block2[0])
        block3 = self.block3(x)
        x = self.pool3(block3[0])
        x = self.conv3(x)
        x = self.conv4(x)

        out1 = self.conv5(x)
        out1 = self.conv6(out1)

        out2 = self.conv7(x)
        out2 = self.up(out2)
        out2 = torch.cat([out2, block3[1]], dim=1)
        out2 = self.conv8(out2)
        out2 = self.conv9(out2)

        return [out2, out1]