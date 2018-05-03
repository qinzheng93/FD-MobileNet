from . import ImageNet

__datasets = ['ImageNet']

def get_dataloader(data_config):
    data_name = data_config['name']
    assert data_name in __datasets, '==> Dataset not supported!'

    if data_name == 'ImageNet':
        train_loader = ImageNet.imagenet_train_loader(
            data_config['train_root'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            transform=data_config['transform']
        )
        valid_loader = ImageNet.imagenet_valid_loader(
            data_config['valid_root'],
            num_workers=data_config['num_workers']
        )
        return train_loader, valid_loader
