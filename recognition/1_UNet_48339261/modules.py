"""
modules.py
"""
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128) 

        # Decoder (upsampling)
        self.dec3 = self._conv_block(128 + 64, 64)
        self.dec2 = self._conv_block(64 + 32, 32)
        self.dec1 = nn.Conv2d(32, out_channels, 1) 

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(e3), e2], 1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1))
        out = self.dec1(d2)

        return out
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        predictions = torch.sigmoid(logits) 

        predictions = predictions.view(-1)
        targets = targets.view(-1).float()

        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice_coeff