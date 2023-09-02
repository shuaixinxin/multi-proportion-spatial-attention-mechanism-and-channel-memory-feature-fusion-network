import torch
import torch.nn as nn



class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SPPAMF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels

        #MaxPool
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * 3, c2, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # AvgPool
        self.cv3 = nn.Conv2d(c1, c_, 1, 1)
        self.cv4 = nn.Conv2d(c_ * 3, c2, 1, 1)
        self.m2 = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

        self.cv5 = nn.Conv2d(c_ * 8, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m1(x1)
        y2 = self.m1(y1)

        x2 = self.cv3(x)
        y3 = self.m2(x2)
        y4 = self.m2(y3)

        z1 = torch.cat([y1, y2, self.m1(y2)], 1)
        z2 = torch.cat([y3, y4, self.m2(y4)], 1)
        z = torch.cat([x, z1, z2], 1)
        out = self.cv5(z)
        return out