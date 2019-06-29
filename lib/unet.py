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


class ConvBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 activ=F.leaky_relu, r=16, se=False):
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(
                in_channels, out_channels, ksize, stride, pad, activ)
            self.conv2 = Conv2DBNActiv(
                out_channels, out_channels, ksize, 1, pad, activ)

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


class SpecUNet(chainer.Chain):

    def __init__(self, pad=(2, 0)):
        super(SpecUNet, self).__init__()
        with self.init_scope():
            self.enc1 = ConvBlock(None, 32, 3, pad=(1, 0), stride=2)
            self.enc2 = ConvBlock(None, 64, 3, pad=(1, 0), stride=2)
            self.enc3 = ConvBlock(None, 128, 3, pad=(1, 0), stride=2)
            self.enc4 = ConvBlock(None, 256, 3, pad=(1, 0), stride=2)
            self.enc5 = ConvBlock(None, 512, 3, pad=(1, 0), stride=2)

            self.bottom = ConvBlock(None, 1024, 3, pad=(1, 0), stride=2)

            self.dec5 = Conv2DBNActiv(None, 512, 3, pad=(1, 0), activ=F.relu)
            self.dec4 = Conv2DBNActiv(None, 256, 3, pad=(1, 0), activ=F.relu)
            self.dec3 = Conv2DBNActiv(None, 128, 3, pad=(1, 0), activ=F.relu)
            self.dec2 = Conv2DBNActiv(None, 64, 3, pad=(1, 0), activ=F.relu)
            self.dec1 = Conv2DBNActiv(None, 32, 3, pad=(1, 0), activ=F.relu)

            self.out = L.Convolution2D(None, 2, 1, nobias=True)

        self.offset = 222

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

        h = F.resize_images(h, (h.shape[2] * 2, h.shape[3] * 2))
        h = F.sigmoid(self.out(spec_utils.crop_and_concat(h, x)))

        return h
