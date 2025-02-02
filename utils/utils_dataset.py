import os.path

from torchvision import datasets
from torchvision.datasets import Flowers102, OxfordIIITPet, FGVCAircraft

from utils.cub200 import CUB200

def get_dataset(dataset_name, transform=None, root_dir='../data'):
    DATASETS = ['cifar10', 'cifar100', 'svhn', 'miniImagenet', 'fmnist', 'cub200', 'flower102', 'pet37', 'fgvc100']
    assert dataset_name in DATASETS
    if dataset_name == 'cifar10':
        return datasets.CIFAR10(root_dir, train=True, download=False, transform=transform)
    elif dataset_name == 'cifar100':
        return datasets.CIFAR100(root_dir, train=True, download=False, transform=transform)
    elif dataset_name == 'cub200':
        return CUB200(root=root_dir, download=False, train=True, w_transform=transform, s_transform=transform, partial_rate=0.000001, noisy_rate=0.000001)
    elif dataset_name == 'flower102':
        return Flowers102(root=os.path.join(root_dir, 'FLOWER102'), split='train', transform=transform, download=True)
    elif dataset_name == 'pet37':
        return OxfordIIITPet(root=root_dir, split='trainval', transform=transform, download=True)
    elif dataset_name == 'fgvc100':
        return FGVCAircraft(root=os.path.join(root_dir, 'FGVC'), split='trainval', transform=transform, download=True)
