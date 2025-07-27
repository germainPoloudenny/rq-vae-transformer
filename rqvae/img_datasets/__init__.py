import os
import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageNet

from .transforms import create_transforms
from rqvae.img_datasets.amplitude3d import Amplitude3D

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


def create_dataset(config, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.dataset, split='train', is_eval=is_eval)
    transforms_val = create_transforms(config.dataset, split='val', is_eval=is_eval)

    root = config.dataset.get('root', None)

    if config.dataset.type == 'amplitude3d':
        root = root if root else 'data/amplitude'
        val_split = config.dataset.get('val_split', 0.1)
        dataset_trn = Amplitude3D(
            root,
            split='train',
            transform=transforms_trn,
            key=config.dataset.get('data_key', 'amplitudes'),
            max_index=config.dataset.get('hkl_max_index', None),
            val_split=val_split,
            pad_to_cube=config.dataset.get('pad_to_cube', False),
            cube_size=config.dataset.get('cube_size', None),
        )
        dataset_val = Amplitude3D(
            root,
            split='val',
            transform=transforms_val,
            key=config.dataset.get('data_key', 'amplitudes'),
            max_index=config.dataset.get('hkl_max_index', None),
            val_split=val_split,
            pad_to_cube=config.dataset.get('pad_to_cube', False),
            cube_size=config.dataset.get('cube_size', None),
        )
    elif config.dataset.type == 'imagenet':
        root = root if root else 'data/imagenet'
        dataset_trn = ImageNet(root, split='train', transform=transforms_trn)
        dataset_val = ImageNet(root, split='val', transform=transforms_val)
    else:
        raise ValueError('%s not supported...' % config.dataset.type)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val = Subset(dataset_val, torch.randperm(len(dataset_val))[:dataset_len])

    if logger is not None:
        logger.info(f'#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}')

    return dataset_trn, dataset_val
