# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# linear-->bn -->relu
from models import wrn16x2


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
        self.n_t = args.n_t  # feature_t 不重复形状的 feature map的个数
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
        """
            args.t_shapes: [[1, 32, 32, 32],[1, 64, 32, 32],[1, 128, 16, 16]]
            t_shape: [1, 32, 32, 32]
            t_shape[1] : 32 --> channel num
            key_layer : [128,128...]
        """
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], args.qk_dim) for t_shape in args.t_shapes])

    def forward(self, g_t):
        """
            g_t: 6 个 [64, 32, 32, 32]
            bs：batch-size: g_t[0].size(0)
            channel_mean: f_t.mean(3).mean(2):  通道个数的每个feature map的平均值 [64, 6, 32]; 6 是 g_t 中 feature map 的个数
            spatial_mean: f_t.pow(2).mean(1): 得各个通道融合的每个feature map [1, 32, 32] ; .view(bs, -1) 再 batch-size 均值化; spatial_mean: [64, 6, 1024]
            query: channel_mean联合对应的query_layer做线性连接，堆叠得到一个, [64, 32] --> query_layer(32, 128) --> [64, 128] --> stack --> [64, 6, 128]
            value: spatial_mean: [64, 6, 1024] 经过 L2 计算得到
        """
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]
        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)], dim=1)
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, value


class LinearTransformStudent(nn.Module):
    def __init__(self, args):
        """
            self.samplers: model_s up or down sample by different teacher size
            bilinear : a bilinear weight
        """
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = args.qk_dim
        self.relu = nn.ReLU(inplace=False)

        self.samplers = nn.ModuleList([Sample(t_shape) for t_shape in args.unique_t_shapes])
        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], self.qk_dim) for s_shape in args.s_shapes])
        self.bilinear = nn_bn_relu(args.qk_dim, args.qk_dim * len(args.t_shapes))

    def forward(self, g_s):
        """
            g_s: 6 个 [64, 32, 32, 32]
            bs：batch-size: g_t[0].size(0)
            channel_mean: [64, 6, 32]; 6 是 g_t 中 feature map 的个数
            spatial_mean: [1, 32, 32] ; .view(bs, -1) ; spatial_mean: [64, 6, 1024] [64, 6, 256] [64, 6, 64]
            bilinear_key: [64, 6, 16]
            value: spatial_mean: [64, 6, 1024] 经过 L2 计算得到
        """
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]
        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)], dim=1).view(bs * self.s, -1)  # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, value


class Sample(nn.Module):
    """
        student layer up or down sample, then global avg pooling, return H * W  vector
        return : torch.Size([64, 6, 1024]) bs:64; num of g_s:6; H * W: 1024
    """

    def __init__(self, t_shape):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        g_s = torch.stack([self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s], dim=1)
        return g_s


if __name__ == '__main__':

    x = torch.randn(64, 3, 32, 32)
    net = wrn16x2(num_classes=100)
    feats, logit = net(x, is_feat=True)
    feats = feats[1:-1]
    print(f"bs: {feats[0].size(0)}\n")

    t_N, t_C, t_H, t_W = [64, 32, 32, 32]
    adp_sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    for i, f in enumerate(feats):
        print(f"f[{i}]:{f.size()};\n")
        adp = adp_sample(f.pow(2).mean(1, keepdim=True)).view(64, -1)
        print(f"adp{i}:{adp.size()};\n")

    sampler = Sample(t_shape=[64, 32, 32, 32])
    g_s = sampler(feats, 64)
    print(g_s.size())
