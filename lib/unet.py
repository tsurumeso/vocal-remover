import torch
from torch import nn
import torch.nn.functional as F

from lib import spec_utils


class CBAM(nn.Module):

    def __init__(self, ch, ratio=16):
        super(CBAM, self).__init__()
        self.sqz = nn.Linear(ch, ch // ratio)
        self.ext = nn.Linear(ch // ratio, ch)
        self.conv = nn.Conv2d(None, 1, 3, 1, 1, bias=False)

    def __call__(self, x, e=None):
        gap = x.mean(dim=(2, 3))
        gmp = x.max(x, dim=(2, 3))
        gap = self.ext(F.relu(self.sqz(gap)))
        gmp = self.ext(F.relu(self.sqz(gmp)))
        x = F.sigmoid(gap + gmp)[:, :, None, None] * x

        gap = x.mean(dim=1)[:, None]
        gmp = x.max(dim=1)[:, None]
        h = self.conv(torch.cat([gap, gmp], dim=1))
        h = F.sigmoid(h) * x

        return h


class Conv2DBNActiv(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 activ=F.relu):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, ksize, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        self.activ = activ

    def __call__(self, x):
        h = self.bn(self.conv(x))

        if self.activ is not None:
            h = self.activ(h)

        return h


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 activ=F.leaky_relu, cbam=False):
        super(Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(
            in_channels, out_channels, ksize, 1, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(
            out_channels, out_channels, ksize, stride, pad, activ=activ)
        self.cbam = CBAM(out_channels) if cbam else None

    def __call__(self, x):
        skip = self.conv1(x)
        h = self.conv2(skip)

        if self.cbam is not None:
            h = self.cbam(h)

        return h, skip


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 cbam=False, dropout=False):
        super(Decoder, self).__init__()
        self.conv = Conv2DBNActiv(in_channels, out_channels, ksize, 1, pad)
        self.cbam = CBAM(out_channels) if cbam else None

        self.dropout = dropout

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = spec_utils.crop_center(x, skip)
        h = self.conv(x)

        if self.cbam is not None:
            h = self.cbam(h)

        if self.dropout:
            h = F.dropout(h)

        return h


class BaseUNet(nn.Module):

    def __init__(self, in_channels, ch, pad):
        super(BaseUNet, self).__init__()
        self.enc1 = Encoder(in_channels, ch, 3, 2, pad)
        self.enc2 = Encoder(ch, ch * 2, 3, 2, pad)
        self.enc3 = Encoder(ch * 2, ch * 4, 3, 2, pad)
        self.enc4 = Encoder(ch * 4, ch * 8, 3, 2, pad)
        self.enc5 = Encoder(ch * 8, ch * 16, 3, 2, pad)
        self.enc6 = Encoder(ch * 16, ch * 32, 3, 2, pad)

        self.dec6 = Decoder(ch * (32 + 32), ch * 32, 3, 1, pad, dropout=True)
        self.dec5 = Decoder(ch * (16 + 32), ch * 16, 3, 1, pad, dropout=True)
        self.dec4 = Decoder(ch * (8 + 16), ch * 8, 3, 1, pad, dropout=True)
        self.dec3 = Decoder(ch * (4 + 8), ch * 4, 3, 1, pad)
        self.dec2 = Decoder(ch * (2 + 4), ch * 2, 3, 1, pad)
        self.dec1 = Decoder(ch * (1 + 2), ch, 3, 1, pad)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)
        h, e5 = self.enc5(h)
        h, e6 = self.enc6(h)

        h = self.dec6(h, e6)
        h = self.dec5(h, e5)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class MultiBandUNet(nn.Module):

    def __init__(self):
        super(MultiBandUNet, self).__init__()
        self.l_band_unet = BaseUNet(3, 8, pad=1)
        self.h_band_unet = BaseUNet(3, 8, pad=1)
        self.full_band_unet = BaseUNet(10, 16, pad=(1, 0))

        self.out = nn.Conv2d(16, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(8, 2, 1, bias=False)

        self.offset = 159

    def __call__(self, x):
        diff = (x[:, 0] - x[:, 1])[:, None]
        aux = torch.cat([x, diff], dim=1)
        bandw = aux.size()[2] // 2
        aux_l = aux[:, :, :bandw]
        aux_h = aux[:, :, bandw:]

        aux_l = self.l_band_unet(aux_l)
        aux_h = self.h_band_unet(aux_h)
        aux = torch.cat([aux_l, aux_h], dim=2)

        h = torch.cat([x, aux], dim=1)
        h = self.full_band_unet(h)

        h = torch.sigmoid(self.out(h))
        aux = torch.sigmoid(self.aux_out(aux))

        return h, aux
