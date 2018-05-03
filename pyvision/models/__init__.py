from . import ImageNet

__datasets = [
    'ImageNet',
]


__imagenet_models = sorted(name for name in ImageNet.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(ImageNet.__dict__[name]))


def __get_imagenet_model(model_config):
    model_name = model_config['name']
    assert model_name in __imagenet_models, '==> Model not supported!'
    return ImageNet.__dict__[model_name](model_config)


def get_model(data_name, model_config):
    assert data_name in __datasets, '==> Dataset not supported!'

    if data_name == 'ImageNet':
        return __get_imagenet_model(model_config)
