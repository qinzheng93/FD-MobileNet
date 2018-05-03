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
parser.add_argument('--checkpoint', dest='checkpoint', required=True, metavar='CHECKPOINT_FILE', help='Checkpoint file')
parser.add_argument('--print-freq', dest='print_freq', default=10, type=int, metavar='N', help='Print frequency (default: 10)')

best_prec1 = 0
last_epoch = -1


def main():
    global args, best_prec1, last_epoch
    args = parser.parse_args()
    with open(args.data_config, 'r') as json_file:
        data_config = json.load(json_file)
    with open(args.model_config, 'r') as json_file:
        model_config = json.load(json_file)
    if not os.path.exists(args.checkpoint):
        raise RuntimeError('checkpoint `{}` does not exist.'.format(args.checkpoint))

    # create model
    print('==> Creating model `{}`...'.format(model_config['name']))
    model = models.get_model(data_config['name'], model_config)
    checkpoint = torch.load(args.checkpoint)
    print('==> Checkpoint name is `{}`.'.format(checkpoint['name']))
    model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.DataParallel(model).cuda()
    print('==> Creating model completed.')

    print('==> Creating dataloaders...')
    train_loader, valid_loader = dataloaders.get_dataloader(data_config)
    print('==> Creating dataloaders completed.')

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()

    validate(valid_loader, model, criterion)


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


if __name__ == '__main__':
    main()
