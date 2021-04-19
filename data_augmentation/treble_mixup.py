import numpy as np
import torch
import torchvision
from torchvision import transforms as transforms
from torch.autograd import Variable
import utils

class treble_mixup(object):
    @staticmethod
    def data(x, y, alpha=1.0, use_cuda=True):
        """
            Returns mixed inputs, pairs of targets, and lambda
            :param x: input 输入
            :param y: target
            :param alpha: 一个超参
            :param use_cuda: 是否使用cuda
            """
        if alpha > 0:
            lam_1 = np.random.beta(alpha, alpha)
            lam_2 = np.random.beta(alpha, alpha) * 0.67
        else:
            lam_1 = 1
            lam_2 = 1

        batch_size = x.size()[0]
        if use_cuda:
            index_1 = torch.randperm(batch_size).cuda()
            index_2 = torch.randperm(batch_size).cuda()
        else:
            index_1 = torch.randperm(batch_size)
            index_2 = torch.randperm(batch_size)

        mixed_x = lam_1 * x + (1 - lam_1) * x[index_1, :]
        mixed_x = lam_2 * x[index_2, :] + (1 - lam_2) * mixed_x

        # 展示mixup图像结果
        # print(f"lam:{lam_1}, {lam_2}")
        # images_show = torch.cat((x, x[index_1], x[index_2], mixed_x), dim=3)
        # utils.imshow(torchvision.utils.make_grid(images_show, pad_value=5))

        # y_a,y_b和y_c分别是三个图片的one-hot标签
        y_a, y_b, y_c = y, y[index_1], y[index_2]
        return mixed_x, y_a, y_b, y_c, lam_1, lam_2

    @staticmethod
    def criterion(criterion, pred, y_a, y_b, y_c, lam_1, lam_2):
        mixed_criterion = lam_1 * criterion(pred, y_a) + (1 - lam_1) * criterion(pred, y_b)
        mixed_criterion = lam_2 * criterion(pred, y_c) + (1 - lam_2) * mixed_criterion
        return mixed_criterion

    @staticmethod
    def correct(predicted, targets_a, targets_b, targets_c, lam_1, lam_2):
        correct = (lam_1 * predicted.eq(targets_a.data).cpu().sum().float() +
                   (1 - lam_1) * predicted.eq(targets_b.data).cpu().sum().float())
        correct = (lam_2 * predicted.eq(targets_c.data).cpu().sum().float() +
                   (1 - lam_2) * correct)
        return correct

    @staticmethod
    def train(inputs, targets, args, use_cuda, net, criterion, train_loss, total, correct):
        inputs, targets_a, targets_b, targets_c, lam_1, lam_2 = treble_mixup.data(inputs, targets, args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        outputs = net(inputs)
        loss = treble_mixup.criterion(criterion, outputs, targets_a, targets_b, targets_c, lam_1, lam_2)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        correct += treble_mixup.correct(predicted, targets_a, targets_b, targets_c, lam_1, lam_2)
        return train_loss, correct, total, loss
