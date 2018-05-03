from __future__ import print_function
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

__all__ = ['imagenet_train_loader', 'imagenet_valid_loader']

__imagenet_mean = (0.485, 0.456, 0.406)
__imagenet_stdv = (0.229, 0.224, 0.225)
__imagenet_eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
__imagenet_eigvec = torch.Tensor([
    [-0.5675,  0.7192,  0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948,  0.4203],
])

__imagenet_transforms = [
    'more_aggressive',
    'more_aggressive_with_color_jitter',
    'aggressive',
    'aggressive_with_color_jitter',
    'less_aggressive',
    'less_aggressive_with_color_jitter',
    'conservative',
    'conservative_with_color_jitter',
]

def imagenet_train_loader(root, batch_size=256, num_workers=4, transform='more_aggressive_with_color_jitter'):
    assert transform in __imagenet_transforms, 'transform `{}` not supported'.format(transform)
    print('==> Using `{}` transform'.format(transform))
    if transform == 'more_aggressive':
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(__imagenet_mean, __imagenet_stdv)
        ])
    elif transform == 'more_aggressive_with_color_jitter':
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(__imagenet_mean, __imagenet_stdv)
        ])
    elif transform == 'aggressive':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.36, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(__imagenet_mean, __imagenet_stdv)
        ])
    elif transform == 'aggressive_with_color_jitter':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.36, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(__imagenet_mean, __imagenet_stdv)
        ])
    elif transform == 'less_aggressive':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.64, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(__imagenet_mean, __imagenet_stdv)
        ])
    elif transform == 'less_aggressive_with_color_jitter':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.64, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(__imagenet_mean, __imagenet_stdv)
        ])
    elif transform == 'conservative':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(__imagenet_mean, __imagenet_stdv)
        ])
    elif transform == 'conservative_with_color_jitter':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(__imagenet_mean, __imagenet_stdv)
        ])
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root, transform_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader

def imagenet_valid_loader(root, batch_size=500, num_workers=4):
    transform_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(__imagenet_mean, __imagenet_stdv)
    ])
    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root, transform_valid),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return valid_loader
