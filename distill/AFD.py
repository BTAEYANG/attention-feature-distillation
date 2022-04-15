# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# linear-->bn -->relu
class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)  # 全连接
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


class AFD(nn.Module):
    def __init__(self, args):
        super(AFD, self).__init__()
        self.guide_layers = args.guide_layers
        self.hint_layers = args.hint_layers
        self.attention = Attention(args)

    def forward(self, g_s, g_t):
        g_t = [g_t[i] for i in self.guide_layers]
        g_s = [g_s[i] for i in self.hint_layers]
        loss = self.attention(g_s, g_t)
        return sum(loss)


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.qk_dim = args.qk_dim  # 128
        self.n_t = args.n_t  # feature_t 不重复形状的feature map的个数
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), args.qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), args.qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = []

        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        """
            args.t_shapes: [[1, 32, 32, 32],[1, 64, 32, 32],[1, 128, 16, 16]]
            t_shape: [1, 32, 32, 32]
            t_shape[1] : 32 --> channel num
            key_layer : [128,128...] 
        """
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)  # bs：batch-size
        # g_t: [1, 32, 32, 32]
        # mean(3).mean(2): [1, 32] 得到的是通道个数的每个feature map的平均值; channel_mean: [1, 32]
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        # pow(2).mean(1): [1, 32, 32] 得各个通道融合的每个feature map; .view(bs, -1) 再 batch-size 均值化; spatial_mean: [1, 1024]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        # 根据得到的每个channel_mean，联合对应层的query_layer线性连接，堆叠得到一个, [1, 32] --> query_layer(32,128) [1, 32] * [32, 128] --> [1, 128]
        # 再叠加每个Feature map的经过query_layer之后的channel_mean
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)], dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]  # 得到L2计算
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)

        """
            model_s up or down sample
        """
        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])

        """
            args.s_shapes: [[1, 32, 32, 32],[1, 64, 32, 32],[1, 128, 16, 16]]
            s_shape: [1, 32, 32, 32]
            s_shape[1] : 32 --> channel num
            key_layer : [128,128...]
        """
        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes])

        """
            bilinear : a bilinear weight (W Q-K t)
        """

        self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                          dim=1).view(bs * self.s, -1)  # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s
