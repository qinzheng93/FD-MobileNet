from __future__ import print_function

import argparse
import os
import shutil
import time
import json

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import pyvision.dataloaders as dataloaders
import pyvision.models as models
import pyvision.optim as optim

parser = argparse.ArgumentParser(description='PyTorch Classifier Training')
parser.add_argument('--data', dest='data_config', required=True, metavar='DATA_CONFIG', help='Dataset config file')
parser.add_argument('--model', dest='model_config', required=True, metavar='MODEL_CONFIG', help='Model config file')
parser.add_argument('--optim', dest='optim_config', required=True, metavar='OPTIM_CONFIG', help='Optimizer config file')
parser.add_argument('--sched', dest='sched_config', required=True, metavar='SCHED_CONFIG', help='Learning rate scheduler config file')
parser.add_argument('--label', dest='label', required=True, metavar='MODEL_LABEL', help='Model label for checkpoint')
parser.add_argument('--print-freq', dest='print_freq', default=10, type=int, metavar='N', help='Print frequency (default: 10)')
parser.add_argument('--resume', dest='resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on validation set')
parser.add_argument('--evaluate_first', dest='evaluate_first', action='store_true', help='Evaluate model when resuming from a checkpoint')

best_prec1 = 0
last_epoch = -1


def main():
    global args, best_prec1, last_epoch
    args = parser.parse_args()
    with open(args.data_config, 'r') as json_file:
        data_config = json.load(json_file)
    with open(args.model_config, 'r') as json_file:
        model_config = json.load(json_file)
    with open(args.optim_config, 'r') as json_file:
        optim_config = json.load(json_file)
    with open(args.sched_config, 'r') as json_file:
        sched_config = json.load(json_file)
    ckpt_filename = os.path.join('checkpoints', args.label + '.pth.tar')
    best_filename = os.path.join('checkpoints', args.label + '_best.pth.tar')
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
        print('==> mkdir `checkpoints`')
    elif not os.path.isdir('checkpoints'):
        raise RuntimeError('`checkpoints` is not a directory')

    # create model
    print('==> Creating model `{}`...'.format(model_config['name']))
    model = models.get_model(data_config['name'], model_config)
    init_params(model, model_config['init_mode'])
    model = torch.nn.DataParallel(model).cuda()
    print('==> Creating model completed.')


    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.get_optimizer(model, optim_config)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(ckpt_filename):
            print('==> Loading checkpoint `{}`...'.format(ckpt_filename))
            checkpoint = torch.load(ckpt_filename)
            last_epoch = checkpoint['last_epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('==> Loading checkpoint completed. (epoch {})'.format(last_epoch))
        else:
            print('==> No checkpoint found at \'{}\'.'.format(ckpt_filename))
            print('==> Training from scratch.')

    scheduler = optim.get_scheduler(optimizer, sched_config, last_epoch)

    cudnn.benchmark = True

    # Data loading code
    print('==> Creating dataloaders...')
    train_loader, valid_loader = dataloaders.get_dataloader(data_config)
    print('==> Creating dataloaders completed.')

    if args.evaluate:
        if data_config['name'] == 'CIFAR-10':
            valid_loader = dataloaders.CIFAR.cifar10_test_loader(
                data_config['root'],
                batch_size=500,
                num_workers=data_config['num_workers']
            )
            ckpt_filename = os.path.join('checkpoints', args.label + '_best.pth.tar')
            print('==> Reloading checkpoint `{}`...'.format(ckpt_filename))
            checkpoint = torch.load(ckpt_filename)
            model.load_state_dict(checkpoint['state_dict'])
        validate(valid_loader, model, criterion)
        return

    if args.evaluate_first:
        prec1 = validate(valid_loader, model, criterion, epoch=last_epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            checkpoint = {
                'model_name': model_config['name'],
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'last_epoch': epoch,
                'best_prec1': best_prec1,
            }
            save_checkpoint(checkpoint, is_best, ckpt_filename, best_filename)

    start_epoch = last_epoch + 1
    end_epoch = optim_config['epochs']
    for epoch in range(start_epoch, end_epoch):
        scheduler.step()

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # save checkpoint
        checkpoint = {
            'model_name': model_config['name'],
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'last_epoch': epoch,
            'best_prec1': best_prec1,
        }
        torch.save(checkpoint, ckpt_filename)

        # evaluate on validation set
        prec1 = validate(valid_loader, model, criterion, epoch=epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            checkpoint['best_prec1'] = best_prec1
            save_checkpoint(checkpoint, is_best, ckpt_filename, best_filename)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('[Train Summary]\t'
          'Epoch: [{0}]\t'
          'Loss {loss.avg: 3f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}'.format(
          epoch, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i + 1, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print('[Valid Summary]\t'
          'Epoch: [{0}]\t'
          'Loss {loss.avg: 3f}\t'
          'Prec@1 {top1.avg:.3f}\t'
          'Prec@5 {top5.avg:.3f}'.format(
          epoch, loss=losses, top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def init_params(net, init_mode='fan_in'):
    '''Initializing network parameters.'''
    assert init_mode in ['fan_in', 'fan_out'], 'init_mode `{}` not supported'.format(init_mode)
    print('==> Initializing weights..')
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode=init_mode)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_uniform(m.weight, mode=init_mode)
            if m.bias is not None:
                init.constant(m.bias, 0)
    print('==> Initializing weights completed.')


if __name__ == '__main__':
    main()
