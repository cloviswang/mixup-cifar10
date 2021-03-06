import numpy as np
import torch
import torchvision
from torchvision import transforms as transforms
from torch.autograd import Variable


class Mixup(object):
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
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]

        # 展示mixup图像结果
        # print(f"lam:{lam}")
        # images_show = torch.cat((x, x[index], mixed_x), dim=3)
        # utils.imshow(torchvision.utils.make_grid(images_show, pad_value=5))

        # y_a和y_b分别是两个图片的one-hot标签
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @staticmethod
    def criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    @staticmethod
    def correct(predicted, targets_a, targets_b, lam):
        correct = (lam * predicted.eq(targets_a.data).cpu().sum().float() +
                   (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        return correct

    @staticmethod
    def train(inputs, targets, args, use_cuda, net, criterion, train_loss, total, correct):
        inputs, targets_a, targets_b, lam = Mixup.data(inputs, targets, args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        outputs = net(inputs)
        loss = Mixup.criterion(criterion, outputs, targets_a, targets_b, lam)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        correct += Mixup.correct(predicted, targets_a, targets_b, lam)
        return train_loss, correct, total, loss
