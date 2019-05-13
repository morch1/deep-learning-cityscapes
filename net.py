import torch
from torch import nn


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.net = DoubleConv2d(in_channels, out_channels)

    def forward(self, x, enc_x):
        x = self.upsample(x)
        x = torch.cat((x, enc_x), dim=1)
        return self.net(x)


class CityscapesNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.enc1 = DoubleConv2d(n_channels, 64)
        self.maxpool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv2d(64, 128)
        self.maxpool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv2d(128, 256)
        self.maxpool3 = nn.MaxPool2d(2)
        self.bridge = DoubleConv2d(256, 512)
        self.dec3 = Decoder(512, 256)
        self.dec2 = Decoder(256, 128)
        self.dec1 = Decoder(128, 64)
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x = self.maxpool1(x1)
        x2 = self.enc2(x)
        x = self.maxpool2(x2)
        x3 = self.enc3(x)
        x = self.maxpool3(x3)
        x = self.bridge(x)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        x = self.out(x)
        return x
