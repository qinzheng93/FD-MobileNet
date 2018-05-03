import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

__all__ = ['get_scheduler']

__schedulers = [
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
]

def get_scheduler(optimizer, sched_config, last_epoch):
    sched_name = sched_config['name']
    assert sched_name in __schedulers, 'scheduler `{}` not supported!'.format(sched_name)
    print('==> Using `{}` learning rate scheduler.'.format(sched_name))

    if sched_name == 'StepLR':
        return lr_scheduler.StepLR(
            optimizer,
            sched_config['step_size'],
            gamma=sched_config['gamma'],
            last_epoch=last_epoch
        )
    elif sched_name == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(
            optimizer,
            sched_config['milestones'],
            gamma=sched_config['gamma'],
            last_epoch=last_epoch
        )
    elif sched_name == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(
            optimizer,
            sched_config['gamma'],
            last_epoch=last_epoch
        )
