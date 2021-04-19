import random

import numpy as np
import torch
import torchvision
from torchvision import transforms as transforms
from torch.autograd import Variable
import utils


class ricap(object):
    @staticmethod
    def data(x, y, alpha=1.0, use_cuda=True):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam_v = np.random.beta(alpha, alpha)
            W = int(lam_v * 32) % 32
            lam_h = np.random.beta(alpha, alpha)
            H = int(lam_h * 32) % 32
        else:
            lam_v = 1
            W = 31
            lam_h = 1
            H = 31

        batch_size = x.size()[0]
        if use_cuda:
            index_1 = torch.randperm(batch_size).cuda()
            index_2 = torch.randperm(batch_size).cuda()
            index_3 = torch.randperm(batch_size).cuda()
        else:
            index_1 = torch.randperm(batch_size)
            index_2 = torch.randperm(batch_size)
            index_3 = torch.randperm(batch_size)

        # 左上x0，右上x1，左下x2，右下x3
        # print(f"lam_v={lam_v}, lam_h={lam_h}, W={W}, H={H}")
        ricap_x = x * 0.5
        rW = random.randint(0, 32 - W)
        rH = random.randint(0, 32 - H)
        # print(f"rW={rW}, rH={rH}")
        ricap_x[:, :, 0:W, 0:H] = x[:, :, rW:(W + rW), rH:(H + rH)]
        rW = random.randint(0, W)
        rH = random.randint(0, H)
        # print(f"rW={rW}, rH={rH}")
        ricap_x[:, :, W:31, H:31] = x[index_3, :, (W - rW):(31 - rW), (H - rH):(31 - rH)]
        rW = random.randint(0, 32 - W)
        rH = random.randint(0, H)
        # print(f"rW_a={rW_a}, rH_a={rH_a}, rW_b={rW_b}, rH_b={rH_b}")
        ricap_x[:, :, 0:W, H:31] = x[index_2, :, rW:(W + rW), (H - rH):(31 - rH)]
        rW = random.randint(0, W)
        rH = random.randint(0, 32 - H)
        # print(f"rW_a={rW_a}, rH_a={rH_a}, rW_b={rW_b}, rH_b={rH_b}")
        ricap_x[:, :, W:31, 0:H] = x[index_1, :, (W - rW):(31 - rW), rH:(H + rH)]

        # 展示ricap图像结果
        # print(f"lam_v={lam_v}, lam_h={lam_h}, W={W}, H={H}")
        # images_show = torch.cat((x, x[index_1], x[index_2], x[index_3], ricap_x), dim=3)
        # utils.imshow(torchvision.utils.make_grid(images_show, pad_value=5))

        y_a, y_b, y_c, y_d = y, y[index_1], y[index_2], y[index_3]
        return ricap_x, y_a, y_b, y_c, y_d, lam_v, lam_h

    @staticmethod
    def criterion(criterion, pred, y_a, y_b, y_c, y_d, lam_v, lam_h):
        top_left_label = lam_v * lam_h * criterion(pred, y_a)
        top_right_label = (1 - lam_v) * lam_h * criterion(pred, y_b)
        bottom_left_label = lam_v * (1 - lam_h) * criterion(pred, y_c)
        bottom_right_label = (1 - lam_v) * (1 - lam_h) * criterion(pred, y_d)

        label = top_left_label + top_right_label + bottom_left_label + bottom_right_label
        return label

    @staticmethod
    def correct(predicted, targets_a, targets_b, targets_c, targets_d, lam_v, lam_h):
        y_a = predicted.eq(targets_a.data).cpu().sum().float()
        y_b = predicted.eq(targets_b.data).cpu().sum().float()
        y_c = predicted.eq(targets_c.data).cpu().sum().float()
        y_d = predicted.eq(targets_d.data).cpu().sum().float()
        top_left_label = lam_v * lam_h * y_a
        top_right_label = (1 - lam_v) * lam_h * y_b
        bottom_left_label = lam_v * (1 - lam_h) * y_c
        bottom_right_label = (1 - lam_v) * (1 - lam_h) * y_d

        correct = top_left_label + top_right_label + bottom_left_label + bottom_right_label
        return correct

    @staticmethod
    def train(inputs, targets, args, use_cuda, net, criterion, train_loss, total, correct):
        inputs, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2 = ricap.data(inputs, targets, args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        outputs = net(inputs)
        loss = ricap.criterion(criterion, outputs, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        correct += ricap.correct(predicted, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2)
        return train_loss, correct, total, loss
