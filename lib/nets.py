import torch
from torch import nn

from lib import layers


class BaseUNet(nn.Module):

    def __init__(self, nin, ch, pad):
        super(BaseUNet, self).__init__()
        self.enc1 = layers.Encoder(nin, ch, 3, 2, pad)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, pad)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, pad)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, pad)
        self.enc5 = layers.Encoder(ch * 8, ch * 16, 3, 2, pad)

        self.center = nn.Sequential(
            layers.Conv2DBNActiv(ch * 16, ch * 32, 3, 1, pad),
            layers.Conv2DBNActiv(ch * 32, ch * 16, 3, 1, pad),
            nn.Dropout2d(0.1),
        )

        self.dec5 = layers.Decoder(ch * (8 + 16), ch * 16, 3, 1, pad)
        self.dec4 = layers.Decoder(ch * (4 + 16), ch * 8, 3, 1, pad)
        self.dec3 = layers.Decoder(ch * (2 + 8), ch * 4, 3, 1, pad)
        self.dec2 = layers.Decoder(ch * (1 + 4), ch * 2, 3, 1, pad)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, pad)

    def __call__(self, x):
        e2, e1 = self.enc1(x)
        e3, _ = self.enc2(e2)
        e4, _ = self.enc3(e3)
        e5, _ = self.enc4(e4)
        e6, _ = self.enc5(e5)

        h, _ = self.center(e6)

        h = self.dec5(h, e5)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class BaseASPPNet(nn.Module):

    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(ch * 8, dilations)

        self.dec4 = layers.Decoder(ch * (4 + 8), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (2 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (1 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        e2, e1 = self.enc1(x)
        e3, _ = self.enc2(e2)
        e4, _ = self.enc3(e3)
        e5, _ = self.enc4(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


class CascadedASPPNet(nn.Module):

    def __init__(self):
        super(CascadedASPPNet, self).__init__()
        self.low_band_net = BaseASPPNet(2, 32, ((2, 4), (4, 8), (8, 16)))
        self.high_band_net = BaseASPPNet(2, 32, ((2, 4), (4, 8), (8, 16)))

        self.bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.full_band_net = BaseASPPNet(16, 32)

        self.out = nn.Sequential(
            layers.Conv2DBNActiv(32, 16, 3, 1, 1),
            nn.Conv2d(16, 2, 1, bias=False))
        self.aux_out = nn.Conv2d(32, 2, 1, bias=False)

        self.offset = 128

    def __call__(self, x):
        bandw = x.size()[2] // 2
        x_l = x[:, :, :bandw]
        x_h = x[:, :, bandw:]

        aux = torch.cat([
            self.low_band_net(x_l),
            self.high_band_net(x_h)
        ], dim=2)

        h = self.bridge(torch.cat([x, aux], dim=1))
        h = self.full_band_net(h)

        h = torch.sigmoid(self.out(h))
        aux = torch.sigmoid(self.aux_out(aux))

        return h, aux

    def predict(self, x):
        bandw = x.size()[2] // 2
        x_l = x[:, :, :bandw]
        x_h = x[:, :, bandw:]

        aux = torch.cat([
            self.low_band_net(x_l),
            self.high_band_net(x_h)
        ], dim=2)

        h = self.bridge(torch.cat([x, aux], dim=1))
        h = self.full_band_net(h)

        h = torch.sigmoid(self.out(h))
        if self.offset > 0:
            h = h[:, :, :, self.offset:-self.offset]
            assert h.size()[3] > 0

        return h
