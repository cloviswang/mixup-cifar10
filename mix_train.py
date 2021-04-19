#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
from data_augmentation.cut_vh_mixup import cut_vh_mixup
from data_augmentation.ricap import ricap
from data_augmentation.augmix import augmix, AugMixDataset
from data_augmentation.treble_mixup import treble_mixup
from data_augmentation.vh_mixup import vh_mixup
from data_augmentation.mixup import mixup
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--no-jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
parser.add_argument('--all-ops', '-all', action='store_true',
                    help='Turn on all operations (+brightness,contrast,color,sharpness).')
parser.add_argument('--mixture-width', default=3, type=int,
                    help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth', default=-1, type=int,
                    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar10_path = '/home/miao/datasets/'
trainset = datasets.CIFAR10(root=cifar10_path, train=True, download=False,
                            transform=transform_train)
trainset = AugMixDataset(args, trainset, preprocess, True)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root=cifar10_path, train=False, download=False,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                            + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()

if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if (-1 < epoch < 50) or (99 < epoch < 125) or (149 < epoch < 175):
            inputs, targets_a, targets_b, lam = mixup.data(inputs, targets, args.alpha, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = net(inputs)
            loss = mixup.criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            inputs, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2 = ricap.data(inputs, targets, args.alpha,
                                                                                          use_cuda)
            inputs, targets_a, targets_b, targets_c, targets_d = map(Variable, (inputs, targets_a, targets_b,
                                                                                targets_c, targets_d))
            outputs = net(inputs)
            loss = ricap.criterion(criterion, outputs, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2)

        # inputs, targets_a, targets_b, lam = mixup.data(inputs, targets, args.alpha, use_cuda)
        # inputs, targets_a, targets_b, targets_c, lam_1, lam_2 = treble_mixup.data(inputs, targets, args.alpha, use_cuda)
        # inputs, targets_a, targets_b, lam_v, lam_h, lam_mixup = vh_mixup.data(inputs, targets, args.alpha, use_cuda)
        # inputs, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2 = ricap.data(inputs, targets, args.alpha, use_cuda)

        # loss = mixup.criterion(criterion, outputs, targets_a, targets_b, lam)
        # loss = treble_mixup.criterion(criterion, outputs, targets_a, targets_b, targets_c, lam_1, lam_2)
        # loss = vh_mixup.criterion(criterion, outputs, targets_a, targets_b, lam_v, lam_h, lam_mixup)
        # loss = ricap.criterion(criterion, outputs, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        if (-1 < epoch < 50) or (99 < epoch < 125) or (149 < epoch < 175):
            correct += mixup.correct(predicted, targets_a, targets_b, lam)
        else:
            correct += ricap.correct(predicted, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2)
        # correct += mixup.correct(predicted, targets_a, targets_b, lam)
        # correct += treble_mixup.correct(predicted, targets_a, targets_b, targets_c, lam_1, lam_2)
        # correct += vh_mixup.correct(predicted, targets_a, targets_b, lam_v, lam_h, lam_mixup)
        # correct += ricap.correct(predicted, targets_a, targets_b, targets_c, targets_d, lam_1, lam_2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 78 == 0:
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), reg_loss / (batch_idx + 1),
                            100. * float(correct) / float(total), correct, total))

    return train_loss / batch_idx, reg_loss / batch_idx, 100. * float(correct) / float(total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 99 == 0:
            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * float(correct) / float(total),
                            correct, total))

    acc = 100. * correct / total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return test_loss / batch_idx, 100. * float(correct) / float(total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7' + args.name + '_'
               + str(args.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])
