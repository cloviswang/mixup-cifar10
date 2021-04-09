import numpy as np
import torch
import torchvision
from torchvision import transforms as transforms

import utils

class vh_mixup(object):
    @staticmethod
    def data(x, y, alpha=1.0, use_cuda=True):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam_mixup = np.random.beta(alpha, alpha)
            lam_v = np.random.beta(alpha, alpha)
            W = int(lam_v * 32) % 32
            lam_h = np.random.beta(alpha, alpha)
            H = int(lam_h * 32) % 32
        else:
            lam_mixup = 1
            lam_v = 1
            W = 31
            lam_h = 1
            H = 31

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        # 左上x1，右下x2，其他mixup(x1,x2)
        hv_mixed_x = lam_mixup * x + (1 - lam_mixup) * x[index, :]
        hv_mixed_x[:, :, 0:W, 0:H] = x[:, :, 0:W, 0:H]
        hv_mixed_x[:, :, W:31, H:31] = x[index, :, W:31, H:31]

        # 展示vh_mixup图像结果
        # print(f"lam_v={lam_v}, lam_h={lam_h}, lam_mixup={lam_mixup}, W={W}, H={H}")
        # images_show = torch.cat((x, x[index], hv_mixed_x), dim=3)
        # utils.imshow(torchvision.utils.make_grid(images_show, pad_value=5))

        y_a, y_b = y, y[index]
        return hv_mixed_x, y_a, y_b, lam_v, lam_h, lam_mixup

    @staticmethod
    def criterion(criterion, pred, y_a, y_b, lam_v, lam_h, lam_mixup):
        top_left_label = lam_v * lam_h * criterion(pred, y_a)
        bottom_right_label = (1 - lam_v) * (1 - lam_h) * criterion(pred, y_b)
        mixup_label = ((1 - lam_v) * lam_h + lam_v * (1 - lam_h)) * \
                      (lam_mixup * criterion(pred, y_a) + (1 - lam_mixup) * criterion(pred, y_b))
        label = top_left_label + bottom_right_label + mixup_label

        return label

    @staticmethod
    def correct(predicted, targets_a, targets_b, lam_v, lam_h, lam_mixup):
        y_a = predicted.eq(targets_a.data).cpu().sum().float()
        y_b = predicted.eq(targets_b.data).cpu().sum().float()
        top_left_label = lam_v * lam_h * y_a
        bottom_right_label = (1 - lam_v) * (1 - lam_h) * y_b
        mixup_label = ((1 - lam_v) * lam_h + lam_v * (1 - lam_h)) * \
                      (lam_mixup * y_a + (1 - lam_mixup) * y_b)
        correct = top_left_label + bottom_right_label + mixup_label
        return correct
