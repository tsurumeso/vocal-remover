import torch
from torch import nn
import torch.nn.functional as F

from lib import layers


class BaseNet(nn.Module):

    def __init__(self, nin, ch, nin_lstm, dilations=(4, 8, 16)):
        super(BaseNet, self).__init__()
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)
        self.lstm_aspp = layers.LSTMModule(ch * 16, nin_lstm)

        self.dec4 = layers.Decoder(ch * (8 + 16) + 1, ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)

        h = self.aspp(h)
        h = torch.cat([h, self.lstm_aspp(h)], dim=1)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)

        return h


# class BaseNet(nn.Module):

#     def __init__(self, nin, ch, dilations=(4, 8, 16)):
#         super(BaseNet, self).__init__()
#         self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
#         self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
#         self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
#         self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)

#         self.aspp = layers.ASPPModule(ch * 8, ch * 16, dilations)

#         self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
#         self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
#         self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
#         self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

#     def __call__(self, x):
#         h, e1 = self.enc1(x)
#         h, e2 = self.enc2(h)
#         h, e3 = self.enc3(h)
#         h, e4 = self.enc4(h)

#         h = self.aspp(h)

#         h = self.dec4(h, e4)
#         h = self.dec3(h, e3)
#         h = self.dec2(h, e2)
#         h = self.dec1(h, e1)

#         return h


class CascadedNet(nn.Module):

    def __init__(self, n_fft):
        super(CascadedNet, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 16
        self.offset = 64

        self.stg1_low_band_net = BaseNet(2, 16, self.nin_lstm // 2)
        self.stg1_high_band_net = BaseNet(2, 16, self.nin_lstm // 2)
        self.stg1_full_band_net = BaseNet(2, 16, self.nin_lstm)

        self.bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg2_full_band_net = BaseNet(16, 32, self.nin_lstm)

        self.out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(32, 2, 1, bias=False)

    def forward(self, x):
        mix = x.detach().clone()
        x = x[:, :, :self.max_bin]

        bandw = x.size()[2] // 2
        aux = torch.cat([
            torch.cat([
                self.stg1_low_band_net(x[:, :, :bandw]),
                self.stg1_high_band_net(x[:, :, bandw:])
            ], dim=2),
            self.stg1_full_band_net(x)
        ], dim=1)

        h = self.bridge(torch.cat([x, aux], dim=1))
        h = self.stg2_full_band_net(h)

        mask = torch.sigmoid(self.out(h))
        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode='replicate'
        )

        if self.training:
            aux = torch.sigmoid(self.aux_out(aux))
            aux = F.pad(
                input=aux,
                pad=(0, 0, 0, self.output_bin - aux.size()[2]),
                mode='replicate'
            )
            return mask * mix, aux * mix
        else:
            return mask * mix

    def predict(self, x_mag):
        h = self.forward(x_mag)

        if self.offset > 0:
            h = h[:, :, :, self.offset:-self.offset]
            assert h.size()[3] > 0

        return h


# class CascadedNet(nn.Module):

#     def __init__(self, n_fft):
#         super(CascadedNet, self).__init__()
#         self.max_bin = n_fft // 2
#         self.output_bin = n_fft // 2 + 1
#         self.nin_lstm = self.max_bin // 16
#         self.offset = 64

#         self.stg1_low_band_net = BaseNet(2, 16, self.nin_lstm // 2)
#         self.stg1_high_band_net = BaseNet(2, 16, self.nin_lstm // 2)

#         self.stg2_bridge = layers.Conv2DBNActiv(18, 8, 1, 1, 0)
#         self.stg2_full_band_net = BaseNet(8, 16, self.nin_lstm)

#         self.stg3_bridge = layers.Conv2DBNActiv(34, 16, 1, 1, 0)
#         self.stg3_full_band_net = BaseNet(16, 32, self.nin_lstm)

#         self.out = nn.Conv2d(32, 2, 1, bias=False)
#         self.aux_out = nn.Conv2d(32, 2, 1, bias=False)

#     def forward(self, x):
#         mix = x.detach().clone()
#         x = x[:, :, :self.max_bin]

#         bandw = x.size()[2] // 2
#         aux1 = torch.cat([
#             self.stg1_low_band_net(x[:, :, :bandw]),
#             self.stg1_high_band_net(x[:, :, bandw:])
#         ], dim=2)

#         h = torch.cat([x, aux1], dim=1)
#         aux2 = self.stg2_full_band_net(self.stg2_bridge(h))

#         h = torch.cat([x, aux1, aux2], dim=1)
#         h = self.stg3_full_band_net(self.stg3_bridge(h))

#         mask = torch.sigmoid(self.out(h))
#         mask = F.pad(
#             input=mask,
#             pad=(0, 0, 0, self.output_bin - mask.size()[2]),
#             mode='replicate'
#         )

#         if self.training:
#             aux = torch.cat([aux1, aux2], dim=1)
#             aux = torch.sigmoid(self.aux_out(aux))
#             aux = F.pad(
#                 input=aux,
#                 pad=(0, 0, 0, self.output_bin - aux.size()[2]),
#                 mode='replicate'
#             )
#             return mask * mix, aux * mix
#         else:
#             return mask * mix

#     def predict(self, x_mag):
#         h = self.forward(x_mag)

#         if self.offset > 0:
#             h = h[:, :, :, self.offset:-self.offset]
#             assert h.size()[3] > 0

#         return h
