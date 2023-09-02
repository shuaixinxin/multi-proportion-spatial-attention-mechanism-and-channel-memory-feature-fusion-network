import cv2
import numpy as np
import torch
import torch.nn as nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k, s=1, p=None, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Conv1X3(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=(1,3), s=1, p=(0,1), act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Conv3X1(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=(3, 1), s=1, p=(1, 0),  act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class C3(nn.Module):
    #  with 3 convolutions
    def __init__(self, c1, c2, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.cv1 = Conv(c1, c2, 3)
        self.cv2 = Conv1X3(c1, c2)
        self.cv3 = Conv3X1(c1, c2)


    def forward(self, x):
        c1 = self.cv1(x)
        c2 = self.cv2(x)
        c3 = self.cv3(x)
        return torch.cat([c1] + [c2] + [c3], 1)

class C5(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super(C5, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(c_, c_, 3)
        #self.cv4 = Conv(c_, c_, 1)
        self.m = C3(c_, c_)
        self.cv5 = Conv(3 * c_, c_, 1)
        self.cv6 = Conv(c_, c_, 3)
        self.cv7 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.cv3(self.cv1(x))
        x2 = self.m(x1)
        y1 = self.cv6(self.cv5(x2))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
