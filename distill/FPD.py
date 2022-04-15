import torch
from torch import nn
import torch.nn.functional as F

from distill.SELayer import SELayer


class FPD(nn.Module):
    """ Feature Pyramid Distilling """

    def __init__(self):
        super(FPD, self).__init__()

        self.top_layer = nn.Conv2d(256, 256, 1, 1, 0)
        # expand channel
        self.lat_layer2 = nn.Conv2d(128, 256, 1, 1, 0)
        self.lat_layer1 = nn.Conv2d(64, 256, 1, 1, 0)
        self.lat_layer0 = nn.Conv2d(32, 256, 1, 1, 0)
        # feature pyramid smooth
        self.smooth = nn.Conv2d(256, 256, 3, 1, 1)

        # reduction = 16
        self.se = SELayer(256, 16)

    @staticmethod
    def _upSample_add(x, y):
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, f_t, f_s):
        # t0-32, t1-64, t2-128, t3-256, t4-256 (out)
        # _upSample_add
        t_p3 = self.top_layer(f_t[3])  # 256
        t_p2 = self._upSample_add(t_p3, self.lat_layer2(f_t[2]))  # 256 + (128 -> 256)
        t_p1 = self._upSample_add(t_p2, self.lat_layer1(f_t[1]))  # 256 + (64 -> 256)
        t_p0 = self._upSample_add(t_p1, self.lat_layer0(f_t[0]))  # 256 + (32 -> 256)

        # f0-32, f1-64, f2-128, f3-256, f4-256 (out)
        # _upSample_add
        s_p3 = self.top_layer(f_s[3])  # 256
        s_p2 = self._upSample_add(s_p3, self.lat_layer2(f_s[2]))  # 256 + (128 -> 256)
        s_p1 = self._upSample_add(s_p2, self.lat_layer1(f_s[1]))  # 256 + (64 -> 256)
        s_p0 = self._upSample_add(s_p1, self.lat_layer0(f_s[0]))  # 256 + (32 -> 256)

        # 在得到相加后的特征后，利用3×3卷积对生成的P1至P3再进行融合，目的是消除上采样过程带来的重叠效应，以生成最终的特征图
        t_p = [t_p3, t_p2, t_p1, t_p0]  # t_p3 256*8*8  t_p2 256*16*16  t_p1 256*32*32  t_p0 256*32*32
        s_p = [s_p3, s_p2, s_p1, s_p0]

        g_t = [self.smooth(t_p[index]) for index, feature in enumerate(t_p)]
        g_s = [self.smooth(s_p[index]) for index, feature in enumerate(s_p)]

        se_g_t = [self.se(g_t[index]) for index, feature in enumerate(g_t)]
        se_g_s = [self.se(g_s[index]) for index, feature in enumerate(g_s)]

        return g_t, g_s, se_g_t, se_g_s


class FPD_Loss(nn.Module):
    """ Feature Pyramid Distilling Loss"""

    def __init__(self):
        super(FPD_Loss, self).__init__()
        # mse loss
        self.mse = nn.MSELoss()

    def forward(self, g_t, g_s, se_g_t, se_g_s):

        mse = [self.mse(g_t[index], g_s[index]) for index, feature in enumerate(g_t)]
        fpd_factor = F.softmax(torch.Tensor(mse), dim=-1)
        loss_f = sum(fpd_factor[index] * mse[index] for index, value in enumerate(mse))

        # se-layer
        se_mse = [self.mse(se_g_t[index], se_g_s[index]) for index, feature in enumerate(se_g_t)]
        se_factor = F.softmax(torch.Tensor(se_mse), dim=-1)
        loss_se_f = sum(se_factor[index] * se_mse[index] for index, value in enumerate(se_mse))

        loss = [loss_f, loss_se_f]
        factor = F.softmax(torch.Tensor(loss), dim=-1)
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))

        return loss_t
