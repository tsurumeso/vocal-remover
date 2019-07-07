import chainer
import chainer.functions as F
import chainer.links as L

from lib import spec_utils


class Conv2DBNActiv(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, activ=F.relu):
        super(Conv2DBNActiv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels, out_channels, ksize, stride, pad, nobias)
            self.bn = L.BatchNormalization(out_channels)

        self.activ = activ

    def __call__(self, x):
        h = self.bn(self.conv(x))

        if self.activ is not None:
            h = self.activ(h)

        return h


class MobileConv2DBNActiv(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, activ=F.relu):
        super(MobileConv2DBNActiv, self).__init__()
        with self.init_scope():
            self.conv1 = L.DepthwiseConvolution2D(
                in_channels, 1, ksize, stride, pad, nobias)
            self.conv2 = L.Convolution2D(
                in_channels, out_channels, 1, 1, 0, nobias)
            self.bn1 = L.BatchNormalization(in_channels, axis=(0, 2, 3))
            self.bn2 = L.BatchNormalization(out_channels)

        self.activ = activ

    def __call__(self, x):
        h = self.bn1(self.conv1(x))

        if self.activ is not None:
            h = self.activ(h)

        h = self.bn2(self.conv2(h))

        if self.activ is not None:
            h = self.activ(h)

        return h


class ConvBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 activ=F.leaky_relu, r=16, se=False):
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(
                in_channels, out_channels, ksize, stride, pad, activ=activ)
            self.conv2 = Conv2DBNActiv(
                out_channels, out_channels, ksize, 1, pad, activ=activ)

            if se:
                self.fc1 = L.Linear(out_channels, out_channels // r)
                self.fc2 = L.Linear(out_channels // r, out_channels)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        if hasattr(self, 'fc1') and hasattr(self, 'fc2'):
            se = F.relu(self.fc1(F.average(h, axis=(2, 3))))
            se = F.sigmoid(self.fc2(se))[:, :, None, None]
            se = F.broadcast_to(se, h.shape)
            h = h * se

        return h


class MobileConvBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 activ=F.leaky_relu, r=16, se=False):
        super(MobileConvBlock, self).__init__()
        with self.init_scope():
            self.conv1 = MobileConv2DBNActiv(
                in_channels, out_channels, ksize, stride, pad, activ=activ)
            self.conv2 = MobileConv2DBNActiv(
                out_channels, out_channels, ksize, 1, pad, activ=activ)

            if se:
                self.fc1 = L.Linear(out_channels, out_channels // r)
                self.fc2 = L.Linear(out_channels // r, out_channels)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)

        if hasattr(self, 'fc1') and hasattr(self, 'fc2'):
            se = F.relu(self.fc1(F.average(h, axis=(2, 3))))
            se = F.sigmoid(self.fc2(se))[:, :, None, None]
            se = F.broadcast_to(se, h.shape)
            h = h * se

        return h


class BaseUNet(chainer.Chain):

    def __init__(self, ch, pad):
        super(BaseUNet, self).__init__()
        with self.init_scope():
            self.enc1 = ConvBlock(None, ch, 3, pad=pad, stride=2)
            self.enc2 = ConvBlock(None, ch * 2, 3, pad=pad, stride=2)
            self.enc3 = ConvBlock(None, ch * 4, 3, pad=pad, stride=2)
            self.enc4 = ConvBlock(None, ch * 8, 3, pad=pad, stride=2)
            self.enc5 = ConvBlock(None, ch * 16, 3, pad=pad, stride=2)

            self.bottom = ConvBlock(None, ch * 16, 3, pad=pad, stride=2)

            self.dec5 = Conv2DBNActiv(None, ch * 16, 3, pad=pad, activ=F.relu)
            self.dec4 = Conv2DBNActiv(None, ch * 8, 3, pad=pad, activ=F.relu)
            self.dec3 = Conv2DBNActiv(None, ch * 4, 3, pad=pad, activ=F.relu)
            self.dec2 = Conv2DBNActiv(None, ch * 2, 3, pad=pad, activ=F.relu)
            self.dec1 = Conv2DBNActiv(None, ch, 3, pad=pad, activ=F.relu)

    def __call__(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)
        h5 = self.enc5(h4)

        h = F.dropout(self.bottom(h5))

        h = F.resize_images(h, (h.shape[2] * 2, h.shape[3] * 2))
        h = self.dec5(spec_utils.crop_and_concat(h, h5))
        h = F.resize_images(h, (h.shape[2] * 2, h.shape[3] * 2))
        h = self.dec4(spec_utils.crop_and_concat(h, h4))
        h = F.resize_images(h, (h.shape[2] * 2, h.shape[3] * 2))
        h = self.dec3(spec_utils.crop_and_concat(h, h3))
        h = F.resize_images(h, (h.shape[2] * 2, h.shape[3] * 2))
        h = self.dec2(spec_utils.crop_and_concat(h, h2))
        h = F.resize_images(h, (h.shape[2] * 2, h.shape[3] * 2))
        h = self.dec1(spec_utils.crop_and_concat(h, h1))

        return h


class MultiBandUNet(chainer.Chain):

    def __init__(self):
        super(MultiBandUNet, self).__init__()
        with self.init_scope():
            self.l_band_unet = BaseUNet(16, pad=(1, 0))
            self.h_band_unet = BaseUNet(16, pad=(1, 0))
            self.full_band_unet = BaseUNet(32, pad=(1, 0))

            self.conv = Conv2DBNActiv(None, 16, 3, pad=(1, 0), activ=F.relu)
            self.out = L.Convolution2D(None, 2, 1, nobias=True)

        self.offset = 223

    def __call__(self, x):
        band_w = 192
        x_l, x_h = x[:, :, :band_w], x[:, :, band_w:]
        h = self.full_band_unet(x)
        h1 = self.l_band_unet(x_l)
        h2 = self.h_band_unet(x_h)
        h = F.concat([h, F.concat([h1, h2], axis=2)])

        h = F.resize_images(h, (h.shape[2] * 2, h.shape[3] * 2))
        h = self.conv(spec_utils.crop_and_concat(h, x))
        h = F.sigmoid(self.out(h))

        return h
