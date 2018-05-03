from __future__ import print_function

import torch
import torch.optim as optim

__all__ = ['get_optimizer']
__optimizers = [
    'SGD',
    'Adadelta',
    'RMSprop',
    'Adam',
]

def get_optimizer(model, optim_config):
    optim_name = optim_config['name']
    assert optim_name in __optimizers, 'optimizer `{}` not supported!'.format(optim_name)
    print('==> Using `{}` optimizer.'.format(optim_name))

    if optim_name == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=optim_config['learning_rate'],
            momentum=optim_config['momentum'],
            weight_decay=optim_config['weight_decay'],
            nesterov=optim_config['nesterov']
        )
    elif optim_name == 'Adadelta':
        return optim.Adadelta(
            model.parameters(),
            lr=optim_config['learning_rate'],
            rho=optim_config['rho'],
            eps=optim_config['epsilon'],
            weight_decay=optim_config['weight_decay']
        )
    elif optim_name == 'RMSprop':
        return optim.RMSprop(
            model.parameters(),
            lr=optim_config['learning_rate'],
            alpha=optim_config['alpha'],
            eps=optim_config['epsilon'],
            weight_decay=optim_config['weight_decay'],
            momentum=optim_config['momentum'],
            centered=optim_config['centered']
        )
    elif optim_name == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=optim_config['learning_rate'],
            betas=tuple(optim_config['betas']),
            eps=optim_config['epsilon'],
            weight_decay=optim_config['weight_decay']
        )
