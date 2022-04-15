# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

import argparse
from utils import *
from dataset import *
from distill import *
import models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from time import time


def str2bool(s):
    if s not in {'F', 'T'}:
        raise ValueError('Not a valid boolean string')
    return s == 'T'


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_dir', default='/home/lab265/lab265/datasets/')
    parser.add_argument('--data', default='CIFAR100')
    parser.add_argument('--trained_dir', default='trained/wrn40x2/model.pth')

    parser.add_argument('--epoch', default=240, type=int)
    parser.add_argument('--schedule', default=[150, 180, 210], type=int, nargs='+')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--alpha', default=0.9, type=float, help='weight for KD (Hinton)')
    parser.add_argument('--temperature', default=4, type=float)

    parser.add_argument('--model_t', default='wrn40x2', type=str)
    parser.add_argument('--model', default='wrn16x2', type=str)

    parser.add_argument('--beta', default=200, type=float)
    parser.add_argument('--qk_dim', default=128, type=int)

    args = parser.parse_args()

    # 根据相应的model取对应的层数
    args.guide_layers = LAYER[args.model_t]
    args.hint_layers = LAYER[args.model]

    for param in sorted(vars(args).keys()):
        print('--{0} {1}'.format(param, vars(args)[param]))

    train_loader, test_loader, args.num_classes, args.image_size = create_loader(args.batch_size, args.data_dir,
                                                                                 args.data)

    model_t = models.__dict__[args.model_t](num_classes=args.num_classes)
    model_t.load_state_dict(torch.load(args.trained_dir))
    model_s = models.__dict__[args.model](num_classes=args.num_classes)
    device = torch.device('cuda')

    # 随机生成2个CIFAR数据集一样大小的噪声image
    data = torch.randn(2, 3, args.image_size, args.image_size)

    model_t.eval()
    model_s.eval()
    with torch.no_grad():
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    # add model_s
    module_list.append(model_s)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    for i, f in enumerate(feat_s):
        print(f"f[{i}]:{f.size()};\n"
              f" f.mean(3).mean(2).size():{f.mean(3).mean(2).size()};\n"
              f"f.pow(2).mean(1).size():{f.pow(2).mean(1).size()}\n;"
              f"f.pow(2).mean(1).view(1, -1):{f.pow(2).mean(1).view(1, -1).size()}\n")

    # 初始化随机生成教师各层和学生各层的feature的形状，存入的是每层的shape [[1, 32, 32, 32],[1, 64, 32, 32],[1, 128, 16, 16]]
    args.s_shapes = [feat_s[i].size() for i in args.hint_layers]
    args.t_shapes = [feat_t[i].size() for i in args.guide_layers]

    for index, s_shape in enumerate(args.s_shapes):
        print(f"s_shape.size():{s_shape.size()}")

    for index, t_shape in enumerate(args.t_shapes):
        print(f"t_shape.size():{t_shape.size()}")

    # teacher net same size feature map 去重, 得到 teacher feature map number 和 不重复的 shape
    args.n_t, args.unique_t_shapes = unique_shape(args.t_shapes)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = DistillKL(args.temperature)
    criterion_kd = AFD(args)

    # add AFD to trainable
    module_list.append(criterion_kd)
    trainable_list.append(criterion_kd)

    criterion = nn.ModuleList([])
    criterion.append(criterion_ce)
    criterion.append(criterion_kl)
    criterion.append(criterion_kd)

    optimizer = optim.SGD(trainable_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # add model_t
    module_list.append(model_t)

    module_list.cuda()
    criterion.cuda()
    cudnn.benchmark = True

    for epoch in range(1, args.epoch + 1):
        s = time()
        adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_acc1, train_acc5 = train_kl(module_list, optimizer, criterion, train_loader, device, args)
        test_acc1, test_acc5 = test(model_s, test_loader, device)
        print(
            'Epoch: {0:>3d} |Train Loss: {1:>2.4f} |Train Top1: {2:.4f} |Train Top5: {3:.4f} |Test Top1: {4:.4f} |Test Top5: {5:.4f}| Time: {6:>5.1f} (s)'
            .format(epoch, train_loss, train_acc1, train_acc5, test_acc1, test_acc5, time() - s))


if __name__ == '__main__':
    main()
