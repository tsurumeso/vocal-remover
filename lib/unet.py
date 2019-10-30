import chainer
import chainer.functions as F
import chainer.links as L

from lib import spec_utils


class CBAM(chainer.Chain):

    def __init__(self, ch, ratio=16):
        super(CBAM, self).__init__()
        with self.init_scope():
            self.sqz = L.Linear(ch, ch // ratio)
            self.ext = L.Linear(ch // ratio, ch)
            self.conv = L.Convolution2D(None, 1, 3, 1, 1, nobias=True)

    def __call__(self, x, e=None):
        gap = F.average(x, axis=(2, 3))
        gmp = F.max(x, axis=(2, 3))
        gap = self.ext(F.relu(self.sqz(gap)))
        gmp = self.ext(F.relu(self.sqz(gmp)))
        x = F.sigmoid(gap + gmp)[:, :, None, None] * x

        gap = F.average(x, axis=1)[:, None]
        gmp = F.max(x, axis=1)[:, None]
        h = self.conv(F.concat([gap, gmp]))
        h = F.sigmoid(h) * x

        return h


class Conv2DBNActiv(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 dropout=False, activ=F.relu):
        super(Conv2DBNActiv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels, out_channels, ksize, stride, pad, nobias=True)
            self.bn = L.BatchNormalization(out_channels)

        self.dropout = dropout
        self.activ = activ

    def __call__(self, x):
        h = self.bn(self.conv(x))

        if self.dropout:
            h = F.dropout(h)

        if self.activ is not None:
            h = self.activ(h)

        return h


class Encoder(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 activ=F.leaky_relu, r=16, cbam=False):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(
                in_channels, out_channels, ksize, 1, pad, activ=activ)
            self.conv2 = Conv2DBNActiv(
                out_channels, out_channels, ksize, stride, pad, activ=activ)

            self.cbam = CBAM(out_channels, r) if cbam else None

    def __call__(self, x):
        h_skip = self.conv1(x)
        h = self.conv2(h_skip)

        if self.cbam is not None:
            h = self.cbam(h)

        return h, h_skip


class Decoder(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 dropout=False):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.conv = Conv2DBNActiv(
                in_channels, out_channels, ksize, 1, pad, dropout=dropout)

    def __call__(self, x, skip=None):
        x = F.resize_images(x, (x.shape[2] * 2, x.shape[3] * 2))
        if skip is not None:
            x = spec_utils.crop_and_concat(x, skip)
        h = self.conv(x)

        return h


class BaseUNet(chainer.Chain):

    def __init__(self, ch, pad):
        super(BaseUNet, self).__init__()
        with self.init_scope():
            self.enc1 = Encoder(None, ch, 3, 2, pad, cbam=False)
            self.enc2 = Encoder(None, ch * 2, 3, 2, pad, cbam=True)
            self.enc3 = Encoder(None, ch * 4, 3, 2, pad, cbam=True)
            self.enc4 = Encoder(None, ch * 8, 3, 2, pad, cbam=True)
            self.enc5 = Encoder(None, ch * 16, 3, 2, pad, cbam=True)
            self.enc6 = Encoder(None, ch * 32, 3, 2, pad, cbam=True)

            self.dec6 = Decoder(None, ch * 32, 3, 1, pad, dropout=True)
            self.dec5 = Decoder(None, ch * 16, 3, 1, pad, dropout=True)
            self.dec4 = Decoder(None, ch * 8, 3, 1, pad, dropout=True)
            self.dec3 = Decoder(None, ch * 4, 3, 1, pad)
            self.dec2 = Decoder(None, ch * 2, 3, 1, pad)
            self.dec1 = Decoder(None, ch, 3, 1, pad)

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


class MultiBandUNet(chainer.Chain):

    def __init__(self, pad=(1, 0)):
        super(MultiBandUNet, self).__init__()
        with self.init_scope():
            self.l_band_unet = BaseUNet(16, pad=pad)
            self.h_band_unet = BaseUNet(16, pad=pad)
            self.full_band_unet = BaseUNet(8, pad=pad)

            self.conv = Conv2DBNActiv(None, 16, 3, pad=pad)
            self.out = L.Convolution2D(None, 2, 1, nobias=True)

        self.offset = 160

    def __call__(self, x):
        bandw = x.shape[2] // 2
        diff = (x[:, 0] - x[:, 1])[:, None]
        x = self.xp.concatenate([x, diff], axis=1)
        x_l, x_h = x[:, :, :bandw], x[:, :, bandw:]
        h1 = self.l_band_unet(x_l)
        h2 = self.h_band_unet(x_h)
        h = self.full_band_unet(x)

        h = self.conv(F.concat([h, F.concat([h1, h2], axis=2)]))
        h = F.sigmoid(self.out(h))

        return h
