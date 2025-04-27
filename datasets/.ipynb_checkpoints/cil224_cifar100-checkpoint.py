import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from typing import Tuple
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
import pandas as pd

from backbone.vit import vit_base_patch16_224_prompt_prototype, VisionTransformer
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from utils import smart_joint
from utils.conf import base_path
from datasets.seq_cub200 import MyCUB200, CUB200, SequentialCUB200
from datasets.seq_tinyimagenet import TinyImagenet, MyTinyImagenet, SequentialTinyImagenet
from datasets.seq_cifar100 import TCIFAR100, MyCIFAR100
from datasets.seq_cifar10 import TCIFAR10, MyCIFAR10
from backbone.ResNetBottleneck import resnet50

class SequentialMutiDomain(ContinualDataset):
    NAME = 'seq-cifar100224'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    
    TINY_MEAN, TINY_STD = [0.4802, 0.4480, 0.3975], [0.2770, 0.2691, 0.2821]
    BIRD_MEAN, BIRD_STD = [0.4758, 0.4685, 0.3870], [0.2376, 0.2282, 0.2475]
    # CIFAR100_MEAN, CIFAR100_STD = [0, 0, 0], [1, 1, 1]
    CIFAR100_MEAN, CIFAR100_STD = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    CIFAR10_MEAN, CIFAR10_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]
    SIZE = (224, 224)

    task_num = -1

    CIFAR100_TRANSFORM = transforms.Compose(
        [transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )
    CIFAR100_TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])    
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        self.task_num += 1

        # Load CIFAR100 datasets
        cifar100_train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True, download=True, transform=self.CIFAR100_TRANSFORM)
        cifar100_test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False, download=True, transform=self.CIFAR100_TEST_TRANSFORM)

        train_dataset = cifar100_train_dataset
        test_dataset = cifar100_test_dataset

        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)

        return train_loader, test_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        # Use the ViT backbone and load pre-trained weights
        model = VisionTransformer(num_classes=200)
        model.load_state_dict(torch.load('vit_model_weights_in21k.pth'))
        model.head = torch.nn.Linear(768, SequentialMutiDomain.N_CLASSES)
        return model
        # return resnet50(SequentialMutiDomain.N_CLASSES_PER_TASK * SequentialMutiDomain.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

