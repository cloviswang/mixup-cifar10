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
from data_augmentation.augmentations import augmentations_all

from data_augmentation.augmix import AugMixDataset

from utils import progress_bar, choose_aug_method

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
parser.add_argument('--aug-method-1', default='mixup', type=str, help='data augmentation method 1')
parser.add_argument('--aug-method-2', default='ricap', type=str, help='data augmentation method 2')
parser.add_argument('--turn-epochs', default=25, type=int, help='每x个epoch交换一次数据增强方法')
parser.add_argument('--augmix', default=False, type=bool, help='是否对训练集做augmix')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset(CIFAR10 or CIFAR100)')

args = parser.parse_args()
if args.dataset == 'CIFAR10':
    num_classes = 10
elif args.dataset == 'CIFAR100':
    num_classes = 100
use_cuda = torch.cuda.is_available()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
cifar10_path = 'D:\github\mixup-cifar10\data'
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
preprocess = transforms.Compose([
    augmentations_all,
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.augmix:
    if args.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root=cifar10_path, train=True, download=False)
    elif args.dataset == 'CIFAR100':
        trainset = datasets.CIFAR100(root=cifar10_path, train=True, download=False)
    trainset = AugMixDataset(args, trainset, preprocess, True)
else:
    if args.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root=cifar10_path, train=True, download=False, transform=transform_train)
    elif args.dataset == 'CIFAR100':
        trainset = datasets.CIFAR100(root=cifar10_path, train=True, download=False, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=0)
if args.dataset == 'CIFAR10':
    testset = datasets.CIFAR10(root=cifar10_path, train=False, download=False, transform=transform_test)
elif args.dataset == 'CIFAR100':
    testset = datasets.CIFAR100(root=cifar10_path, train=False, download=False, transform=transform_test)

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
    net = models.__dict__[args.model](num_classes)

if not os.path.isdir('results'):
    os.mkdir('results')
logname = f'results/log_{net.__class__.__name__}_{args.name}_{args.dataset}_'
if args.augmix:
    logname += "augmix_"
logname += f"({args.aug_method_1}+{args.aug_method_2})_" + str(args.turn_epochs)
logname += '.csv'

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

        if int(epoch / args.turn_epochs) % 2 == 0:
            train_loss, correct, total, loss = choose_aug_method(args.aug_method_1). \
                train(inputs, targets, args, use_cuda, net, criterion, train_loss, total, correct)
        else:
            train_loss, correct, total, loss = choose_aug_method(args.aug_method_2). \
                train(inputs, targets, args, use_cuda, net, criterion, train_loss, total, correct)

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

    acc = 100. * float(correct) / float(total)
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
    with open(logname, 'a') as logfile:
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
