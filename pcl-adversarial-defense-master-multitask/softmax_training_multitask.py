"""
Created on Wed Jan 23 10:15:27 2019

@author: aamir-mustafa
This is Part 1 file for replicating the results for Paper: 
    "Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks"
Here a ResNet model is trained with Softmax Loss for 164 epochs.
"""

# Essential Imports
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utils import AverageMeter, Logger
from resnet_model_multitask import *  # Imports the ResNet Model
from cutout import Cutout
from autoaugment import ImageNetPolicy, CIFAR10Policy


parser = argparse.ArgumentParser("Softmax Training for CIFAR-10 Dataset")
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122, 140],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max-epoch', type=int, default=164)
parser.add_argument('--t-max', type=int, default=164)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')  # gpu to be used
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--filename', type=str, default='robust_model.pth.tar')  # gpu to be used
parser.add_argument('--save-filename', type=str, default='Softmax')  # gpu to be used

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def prevent_overflow(output):  # -----------------------------
    max_output, _ = output.topk(1, 1, True, True)
    output -= max_output.float()
    return output


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + 'CIFAR-10_OnlySoftmax' + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # Data Loading
    num_classes = 10
    print('==> Preparing dataset ')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
        # Cutout(1, 16),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, pin_memory=True,
                                              shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, pin_memory=True,
                                             shuffle=False, num_workers=1)  # args.workers)

    # Loading the Model

    model = frize_resnet(num_classes=num_classes, depth=110, filename=args.filename)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    #######################################################################
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=1e-5)
    start_time = time.time()
    for epoch in range(args.max_epoch):
        # adjust_learning_rate(optimizer, epoch)

        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        # print('LR: %f' % (state['lr']))
        print('LR: %f' % (model_lr.get_lr()[-1]))

        train(trainloader, model, criterion, optimizer, epoch, use_gpu, num_classes, model_lr)

        if args.eval_freq > 0 and (epoch + 0) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")  # Tests after every 10 epochs
            acc128, acc256_1, acc256_2, acc256_3, acc256_4, acc256_5, acc256_6, acc256_7, acc256_8, acc256_9, \
            acc256_10, acc256_11, acc256_12, acc256_13, acc256_14, acc256_15, acc256_16, acc256_17, acc256_18, \
            acc, err, total = test(model, testloader, use_gpu, num_classes, epoch)
            print("Accuracy128 (%): {}\t Accuracy256_1 (%): {}\t Accuracy256_2 (%): {}\t  Accuracy256_3 (%): {}\n"
                  "Accuracy256_4 (%): {}\t Accuracy256_5 (%): {}\t Accuracy256_6 (%): {}\t  Accuracy256_7 (%): {}\n "
                  "Accuracy256_8 (%): {}\t Accuracy256_9 (%): {}\t Accuracy256_10 (%): {}\t  Accuracy256_11 (%): {}\n "
                  "Accuracy256_12 (%): {}\t Accuracy256_13 (%): {}\t Accuracy256_14 (%): {}\t  Accuracy256_15 (%): {}\n "
                  "Accuracy256_16 (%): {}\t Accuracy256_17 (%): {}\t Accuracy256_18 (%): {}\t  Accuracy (%): {}\t "
                  "\t Error rate (%): {} Total {}".format(acc128, acc256_1, acc256_2, acc256_3, acc256_4, acc256_5,
                                                          acc256_6, acc256_7, acc256_8, acc256_9, acc256_10, acc256_11,
                                                          acc256_12, acc256_13, acc256_14, acc256_15, acc256_16,
                                                          acc256_17, acc256_18, acc, err, total))

            checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                          'optimizer_model': optimizer.state_dict(), }
            torch.save(checkpoint, 'Models_Softmax_Multitask/frize_' + str(args.save_filename) + '.pth.tar')

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(trainloader, model, criterion, optimizer, epoch, use_gpu, num_classes, model_lr):
    model.train()

    losses256_15 = AverageMeter()  # 15
    losses256_16 = AverageMeter()  # 16
    losses256_17 = AverageMeter()  # 17
    losses256_18 = AverageMeter()
    losses_outputs = AverageMeter()
    losses = AverageMeter()

    # Batch-wise Training
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        f128, f256_15, f256_16, f256_17, f256_18, f256, f1024,\
        outputs128, outputs256_1, outputs256_2, outputs256_3, outputs256_4, outputs256_5, outputs256_6, outputs256_7, \
        outputs256_8, outputs256_9, outputs256_10, outputs256_11, outputs256_12, outputs256_13, outputs256_14, \
        outputs256_15, outputs256_16, outputs256_17, outputs256_18, outputs = model(data)
        # print('outputs256_18.shape:',outputs256_18.requires_grad)

        outputs256_15 = prevent_overflow(outputs256_15)  # 15
        outputs256_16 = prevent_overflow(outputs256_16)  # 16
        outputs256_17 = prevent_overflow(outputs256_17)  # 17
        outputs256_18 = prevent_overflow(outputs256_18)  # 18
        outputs = prevent_overflow(outputs)

        loss_xent256_15 = criterion(outputs256_15, labels)  # 15
        loss_xent256_16 = criterion(outputs256_16, labels)  # 16
        loss_xent256_17 = criterion(outputs256_17, labels)  # 17
        loss_xent256_18 = criterion(outputs256_18, labels)  # 18
        loss_xent_outputs = criterion(outputs, labels)

        loss_xent = loss_xent256_15 + loss_xent256_16 + loss_xent256_17 + loss_xent256_18 + loss_xent_outputs

        optimizer.zero_grad()
        loss_xent.backward()
        optimizer.step()

        losses256_15.update(loss_xent256_15.item(), labels.size(0))  # 15
        losses256_16.update(loss_xent256_16.item(), labels.size(0))  # 16
        losses256_17.update(loss_xent256_17.item(), labels.size(0))  # 17
        losses256_18.update(loss_xent256_18.item(), labels.size(0))  # 18
        losses_outputs.update(loss_xent_outputs.item(), labels.size(0))
        losses.update(loss_xent.item(), labels.size(0))  # AverageMeter() has this param

        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.3f} ({:.3f}) Loss256_15 {:.3f} ({:.3f}) Loss256_16 {:.3f} ({:.3f})  "
                  "Loss256_17 {:.3f} ({:.3f})  Loss256_18 {:.3f} ({:.3f}) Loss_outputs {:.3f} ({:.3f})  " \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, losses256_15.val, losses256_15.avg,
                          losses256_16.val, losses256_16.avg, losses256_17.val, losses256_17.avg, losses256_18.val,
                          losses256_18.avg,
                          losses_outputs.val, losses_outputs.avg))
    model_lr.step()  # ##################


def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct256_15 = 0
    correct256_16 = 0
    correct256_17 = 0
    correct256_18 = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            f128, f256_15, f256_16, f256_17, f256_18, f256, f1024,\
            outputs128, outputs256_1, outputs256_2, outputs256_3, outputs256_4, outputs256_5, outputs256_6, outputs256_7, \
            outputs256_8, outputs256_9, outputs256_10, outputs256_11, outputs256_12, outputs256_13, outputs256_14, \
            outputs256_15, outputs256_16, outputs256_17, outputs256_18, outputs = model(data)

            outputs256_15 = prevent_overflow(outputs256_15)  # ############
            outputs256_16 = prevent_overflow(outputs256_16)
            outputs256_17 = prevent_overflow(outputs256_17)  # ############
            outputs256_18 = prevent_overflow(outputs256_18)
            outputs = prevent_overflow(outputs)

            predictions256_15 = outputs256_15.data.max(1)[1]
            predictions256_16 = outputs256_16.data.max(1)[1]
            predictions256_17 = outputs256_17.data.max(1)[1]
            predictions256_18 = outputs256_18.data.max(1)[1]
            predictions = outputs.data.max(1)[1]

            total += labels.size(0)
            correct256_15 += (predictions256_15 == labels.data).sum()
            correct256_16 += (predictions256_16 == labels.data).sum()
            correct256_17 += (predictions256_17 == labels.data).sum()
            correct256_18 += (predictions256_18 == labels.data).sum()
            correct += (predictions == labels.data).sum()

    acc128 = 0
    acc256_1 = 0
    acc256_2 = 0
    acc256_3 = 0
    acc256_4 = 0
    acc256_5 = 0
    acc256_6 = 0
    acc256_7 = 0
    acc256_8 = 0
    acc256_9 = 0
    acc256_10 = 0
    acc256_11 = 0
    acc256_12 = 0
    acc256_13 = 0
    acc256_14 = 0
    acc256_15 = correct256_15 * 100. / total
    acc256_16 = correct256_16 * 100. / total
    acc256_17 = correct256_17 * 100. / total
    acc256_18 = correct256_18 * 100. / total
    acc = correct * 100. / total

    err = 100. - acc
    return acc128, acc256_1, acc256_2, acc256_3, acc256_4, acc256_5, acc256_6, acc256_7, acc256_8, \
           acc256_9, acc256_10, acc256_11, acc256_12, acc256_13, acc256_14, acc256_15, acc256_16, \
           acc256_17, acc256_18, acc, err, total


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
