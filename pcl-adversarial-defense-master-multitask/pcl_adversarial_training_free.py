"""
reated on Wed Jan 23 10:15:27 2019

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
from proximity import Proximity
from contrastive_proximity import Con_Proximity
from resnet_model import *  # Imports the ResNet Model
from cutout import Cutout
from autoaugment import ImageNetPolicy, CIFAR10Policy

parser = argparse.ArgumentParser("Softmax Training for CIFAR-10 Dataset")
parser.add_argument('-j', '--workers', default=8, type=int, help="number of data loading workers (default: 2)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N', help='test batchsize')
# parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--lr_model', type=float, default=0.01, help="learning rate for CE Loss")
parser.add_argument('--lr_prox', type=float, default=0.5, help="learning rate for Proximity Loss")  # as per paper
parser.add_argument('--lr_conprox', type=float, default=0.0001,
                    help="learning rate for Con-Proximity Loss")  # as per paper
parser.add_argument('--weight-prox', type=float, default= 1, help="weight for Proximity Loss")  # as per paper
parser.add_argument('--weight-conprox', type=float, default=0.0001,
                    help="weight for Con-Proximity Loss")  # as per paper
parser.add_argument('--clip_eps', type=float, default=4.0 / 255.0, help="FGSM parameters during training")  # #######
parser.add_argument('--fgsm_step', type=float, default=4.0 / 255.0, help="# FGSM parameters during training")  # #######
parser.add_argument('--schedule', type=int, nargs='+', default=[142*781*0.5, 230*781*0.5, 360*781*0.5],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--n-repeats', type=int, default=2)  # #######
parser.add_argument('--max-epoch', type=int, default=500)
# parser.add_argument('--t-max', type=int, default=20000)  ################

parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='1')  # gpu to be used
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--model_name', type=str, default='None')
parser.add_argument('--save_name', type=str, default='None')

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

global total_iter
total_iter = 0


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
        Cutout(1, 16),])
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
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
                                             shuffle=False, num_workers=args.workers)

    # Loading the Model

    model = resnet(num_classes=num_classes, depth=110)

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    criterion_xent = nn.CrossEntropyLoss()
    criterion_prox_1024 = Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_prox_256 = Proximity(num_classes=num_classes, feat_dim=256, use_gpu=use_gpu)

    criterion_conprox_1024 = Con_Proximity(num_classes=num_classes, feat_dim=1024, use_gpu=use_gpu)
    criterion_conprox_256 = Con_Proximity(num_classes=num_classes, feat_dim=256, use_gpu=use_gpu)

    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=1e-04, momentum=0.9)

    optimizer_prox_1024 = torch.optim.SGD(criterion_prox_1024.parameters(), lr=args.lr_prox)
    optimizer_prox_256 = torch.optim.SGD(criterion_prox_256.parameters(), lr=args.lr_prox)

    optimizer_conprox_1024 = torch.optim.SGD(criterion_conprox_1024.parameters(), lr=args.lr_conprox)
    optimizer_conprox_256 = torch.optim.SGD(criterion_conprox_256.parameters(), lr=args.lr_conprox)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
    # weight_decay=args.weight_decay)

    # model_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=1e-5)

    filename = args.model_name  # 'Models_Softmax/CIFAR10_Softmax.pth.tar'
    checkpoint = torch.load(filename)  #

    model.load_state_dict(checkpoint['state_dict'])  #
    optimizer_model.load_state_dict = checkpoint['optimizer_model']  #
    start_time = time.time()
    args.max_epoch = int(500*781*0.5 * args.train_batch / (50000 * args.n_repeats) + 1)

    for epoch in range(args.max_epoch):

        #  adjust_learning_rate(optimizer, epoch)
        # model_lr.step()
        # if total_iter > args.t_max:
        #    break
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print('LR: %f' % (state['lr_model']))

        # print('LR: %f' % (model_lr.get_lr()[-1]))
        train(model, criterion_xent, criterion_prox_1024, criterion_prox_256,
              criterion_conprox_1024, criterion_conprox_256,
              optimizer_model, optimizer_prox_1024, optimizer_prox_256,
              optimizer_conprox_1024, optimizer_conprox_256,
              trainloader, use_gpu, num_classes, epoch)
        # train(trainloader, model, criterion, optimizer, epoch, use_gpu, num_classes, model_lr)

        # if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
        if args.eval_freq > 0 :
            print("==> Test")  # Tests after every 10 epochs
            acc, err = test(model, testloader, use_gpu, num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            if acc > 88 and (epoch + 1) > args.max_epoch - 100:
                state_ = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                          'optimizer_model': optimizer_model.state_dict(),
                          'optimizer_prox_1024': optimizer_prox_1024.state_dict(),
                          'optimizer_prox_256': optimizer_prox_256.state_dict(),
                          'optimizer_conprox_1024': optimizer_conprox_1024.state_dict(),
                          'optimizer_conprox_256': optimizer_conprox_256.state_dict(), }

                torch.save(state_, 'PCL_Models_Adversarial_training_free/CIFAR10_PCL_'+str(args.n_repeats)+'_'+str(args.lr_conprox)+'_' + str(epoch + 1) + '_' + str(float(acc)) + '.pth.tar')

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


# Free Adversarial Training Module
global global_noise_data
global_noise_data = torch.zeros([args.train_batch, 3, 32, 32]).cuda()


def fgsm(gradz, step_size):
    return step_size * torch.sign(gradz)


def train(model, criterion_xent, criterion_prox_1024, criterion_prox_256,
          criterion_conprox_1024, criterion_conprox_256,
          optimizer_model, optimizer_prox_1024, optimizer_prox_256,
          optimizer_conprox_1024, optimizer_conprox_256,
          trainloader, use_gpu, num_classes, epoch):

    global global_noise_data
    data_mean = torch.Tensor(np.array(mean)[:, np.newaxis, np.newaxis])
    data_mean = data_mean.expand(3, 32, 32).cuda()
    data_std = torch.Tensor(np.array(std)[:, np.newaxis, np.newaxis])
    data_std = data_std.expand(3, 32, 32).cuda()
    model.train()
    xent_losses = AverageMeter()  # Computes and stores the average and current value
    prox_losses_1024 = AverageMeter()
    prox_losses_256 = AverageMeter()
    conprox_losses_1024 = AverageMeter()
    conprox_losses_256 = AverageMeter()
    losses = AverageMeter()

    # Batch-wise Training
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        for j in range(args.n_repeats):
            global total_iter
            adjust_learning_rate(optimizer_model, total_iter)
            adjust_learning_rate_prox(optimizer_prox_1024, total_iter)
            adjust_learning_rate_prox(optimizer_prox_256, total_iter)

            adjust_learning_rate_conprox(optimizer_conprox_1024, total_iter)
            adjust_learning_rate_conprox(optimizer_conprox_256, total_iter)
            # Ascend on the global noise
            noise_batch = torch.autograd.Variable(global_noise_data[0:data.size(0)], requires_grad=True).cuda()
            in1 = data + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(data_mean).div_(data_std)
            #true_labels_adv=labels
            #datas=torch.cat((data, in1),0)
            #labels=torch.cat((labels,true_labels_adv))
            feats128, feats256, feats1024, outputs, pre_features = model(in1)
            loss_xent = criterion_xent(outputs, labels)

            loss_prox_1024 = criterion_prox_1024(feats1024, labels)
            loss_prox_256 = criterion_prox_256(feats256, labels)

            loss_conprox_1024 = criterion_conprox_1024(feats1024, labels)
            loss_conprox_256 = criterion_conprox_256(feats256, labels)

            loss_prox_1024 *= args.weight_prox
            loss_prox_256 *= args.weight_prox

            loss_conprox_1024 *= args.weight_conprox
            loss_conprox_256 *= args.weight_conprox
            if args.weight_conprox !=0:
                loss = loss_xent + loss_prox_1024 + loss_prox_256 - loss_conprox_1024 - loss_conprox_256  # total loss
            else:
                loss = loss_xent + 0.0*loss_prox_1024 + 0.0*loss_prox_256 - 0.0*loss_conprox_1024 - 0.0*loss_conprox_256  # total loss
            optimizer_model.zero_grad()

            optimizer_prox_1024.zero_grad()
            optimizer_prox_256.zero_grad()

            optimizer_conprox_1024.zero_grad()
            optimizer_conprox_256.zero_grad()

            loss.backward()
            optimizer_model.step()
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, args.fgsm_step)
            global_noise_data[0:data.size(0)] += pert.data
            global_noise_data.clamp_(-args.clip_eps, args.clip_eps)
            if args.weight_conprox != 0 : #j == args.n_repeats-1:
                for param in criterion_prox_1024.parameters():
                    param.grad.data *= (1. / args.weight_prox)
                optimizer_prox_1024.step()

                for param in criterion_prox_256.parameters():
                    param.grad.data *= (1. / args.weight_prox)
                optimizer_prox_256.step()

                for param in criterion_conprox_1024.parameters():
                    param.grad.data *= (1. / args.weight_conprox)
                optimizer_conprox_1024.step()

                for param in criterion_conprox_256.parameters():
                    param.grad.data *= (1. / args.weight_conprox)
                optimizer_conprox_256.step()


            total_iter += 1
            # model_lr.step()  # ################## big change
            losses.update(loss.item(), labels.size(0))
            xent_losses.update(loss_xent.item(), labels.size(0))
            prox_losses_1024.update(loss_prox_1024.item(), labels.size(0))
            prox_losses_256.update(loss_prox_256.item(), labels.size(0))

            conprox_losses_1024.update(loss_conprox_1024.item(), labels.size(0))
            conprox_losses_256.update(loss_conprox_256.item(), labels.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            # print('total iter : %f' % total_iter)
            # print('LR: %f' % (state['lr_model']))
            print("Batch {}/{}\t Loss {:.3f} ({:.3f})  XentLoss {:.3f} ({:.3f})  ProxLoss_1024 {:.3f} ({:.3f}) "
                "ProxLoss_256 {:.3f} ({:.3f})  ConProxLoss_1024 {:.3f} ({:.3f}) ConProxLoss_256 {:.3f} ({:.3f}) " \
                    .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg,
                            prox_losses_1024.val, prox_losses_1024.avg, prox_losses_256.val, prox_losses_256.avg,
                            conprox_losses_1024.val, conprox_losses_1024.avg, conprox_losses_256.val,
                            conprox_losses_256.avg))


def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            feats128, feats256, feats1024, outputs, pre_features = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_model'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_model'] = state['lr_model']


def adjust_learning_rate_prox(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_prox'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_prox'] = state['lr_prox']


def adjust_learning_rate_conprox(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_conprox'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_conprox'] = state['lr_conprox']


if __name__ == '__main__':
    main()

