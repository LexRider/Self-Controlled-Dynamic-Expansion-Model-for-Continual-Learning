# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import logging
import sys
from typing import List, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as scheds
from torch.utils.data import DataLoader, Dataset

from datasets.utils.validation import get_validation_indexes
from utils.conf import create_seeded_dataloader
from datasets.utils import DEFAULT_ARGS
from utils.prompt_templates import templates


class ContinualDataset(object):
    """
    A base class for defining continual learning datasets.

    Attributes:
        NAME (str): the name of the dataset
        SETTING (str): the setting of the dataset
        N_CLASSES_PER_TASK (int): the number of classes per task
        N_TASKS (int): the number of tasks
        N_CLASSES (int): the number of classes
        SIZE (Tuple[int]): the size of the dataset
        AVAIL_SCHEDS (List[str]): the available schedulers
        class_names (List[str]): list of the class names of the dataset (should be populated by `get_class_names`)
        train_loader (DataLoader): the training loader
        test_loaders (List[DataLoader]): the test loaders
        i (int): the current task
        c_task (int): the current task
        args (Namespace): the arguments which contains the hyperparameters
    """

    NAME: str
    SETTING: str
    N_CLASSES_PER_TASK: int
    N_TASKS: int
    N_CLASSES: int
    SIZE: Tuple[int]
    AVAIL_SCHEDS = ['multisteplr']
    class_names: List[str] = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.

        Args:
            args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.c_task = -1
        self.args = args
        if self.SETTING == 'class-il':
            self.N_CLASSES = self.N_CLASSES if hasattr(self, 'N_CLASSES') else \
                (self.N_CLASSES_PER_TASK * self.N_TASKS) if isinstance(self.N_CLASSES_PER_TASK, int) else sum(self.N_CLASSES_PER_TASK)
        else:
            self.N_CLASSES = self.N_CLASSES_PER_TASK

        if self.args.permute_classes:
            if not hasattr(self.args, 'class_order'):  # set only once
                if self.args.seed is not None:
                    np.random.seed(self.args.seed)
                self.args.class_order = np.random.permutation(self.N_CLASSES)

        if args.joint:
            assert self.SETTING in ['class-il', 'task-il'], 'Joint training is only supported for class-il and task'
            self.N_CLASSES_PER_TASK = self.N_CLASSES
            self.N_TASKS = 1

        if not all((self.NAME, self.SETTING, self.N_CLASSES_PER_TASK, self.N_TASKS, self.SIZE, self.N_CLASSES)):
            raise NotImplementedError('The dataset must be initialized with all the required fields.')

    def update_default_args(self):
        """
        Updates the default arguments with the ones specified in the dataset class.
        Default arguments are defined in the DEFAULT_ARGS dictionary and set by the 'set_default_from_args' decorator.

        Returns:
            Namespace: the updated arguments
        """

        if self.args.dataset not in DEFAULT_ARGS:  # no default args for this dataset
            return self.args

        for k, v in DEFAULT_ARGS[self.args.dataset].items():
            assert hasattr(self.args, k), f'Argument {k} set by the `set_default_from_args` decorator is not present in the arguments.'

            if getattr(self.args, k) is None:
                setattr(self.args, k, v)
            else:
                if getattr(self.args, k) != v:
                    logging.warning('{} set to {} instead of {}.'.format(k, getattr(self.args, k), v))

        return self.args

    def get_offsets(self, task_idx: int = None):
        """
        Compute the start and end class index for the current task.

        Args:
            task_idx (int): the task index

        Returns:
            tuple: the start and end class index for the current task
        """
        if self.SETTING == 'class-il' or self.SETTING == 'task-il':
            task_idx = task_idx if task_idx is not None else self.c_task
        else:
            task_idx = 0

        start_c = self.N_CLASSES_PER_TASK * task_idx if isinstance(self.N_CLASSES_PER_TASK, int) else sum(self.N_CLASSES_PER_TASK[:task_idx])
        end_c = self.N_CLASSES_PER_TASK * (task_idx + 1) if isinstance(self.N_CLASSES_PER_TASK, int) else sum(self.N_CLASSES_PER_TASK[:task_idx + 1])

        assert end_c > start_c, 'End class index must be greater than start class index.'

        return start_c, end_c



    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        raise NotImplementedError

    @staticmethod
    def get_backbone() -> nn.Module:
        """Returns the backbone to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_transform() -> nn.Module:
        """Returns the transform to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_loss() -> nn.Module:
        """Returns the loss to be used for the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_normalization_transform() -> nn.Module:
        """Returns the transform used for normalizing the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_denormalization_transform() -> nn.Module:
        """Returns the transform used for denormalizing the current dataset."""
        raise NotImplementedError

    @staticmethod
    def get_scheduler(model, args: Namespace, reload_optim=True) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for the current dataset.
        If `reload_optim` is True, the optimizer is reloaded from the model. This should be done at least ONCE every task
        to ensure that the learning rate is reset to the initial value.
        """
        if args.lr_scheduler is not None:
            if reload_optim or not hasattr(model, 'opt'):
                model.opt = model.get_optimizer()
            # check if lr_scheduler is in torch.optim.lr_scheduler
            supported_scheds = {sched_name.lower(): sched_name for sched_name in dir(scheds) if sched_name.lower() in ContinualDataset.AVAIL_SCHEDS}
            sched = None
            if args.lr_scheduler.lower() in supported_scheds:
                if args.lr_scheduler.lower() == 'multisteplr':
                    assert args.lr_milestones is not None, 'MultiStepLR requires `--lr_milestones`'
                    sched = getattr(scheds, supported_scheds[args.lr_scheduler.lower()])(model.opt,
                                                                                         milestones=args.lr_milestones,
                                                                                         gamma=args.sched_multistep_lr_gamma)

            if sched is None:
                raise ValueError('Unknown scheduler: {}'.format(args.lr_scheduler))
            return sched
        return None

    def get_iters(self):
        """Returns the number of iterations to be used for the current dataset."""
        raise NotImplementedError('The dataset does not implement the method `get_iters` to set the default number of iterations.')

    def get_epochs(self):
        """Returns the number of epochs to be used for the current dataset."""
        raise NotImplementedError('The dataset does not implement the method `get_epochs` to set the default number of epochs.')

    def get_batch_size(self):
        """Returns the batch size to be used for the current dataset."""
        raise NotImplementedError('The dataset does not implement the method `get_batch_size` to set the default batch size.')

    def get_minibatch_size(self):
        """Returns the minibatch size to be used for the current dataset."""
        return self.get_batch_size()

    def get_class_names(self) -> List[str]:
        """Returns the class names for the current dataset."""
        raise NotImplementedError('The dataset does not implement the method `get_class_names` to get the class names.')

    def get_prompt_templates(self) -> List[str]:
        """
        Returns the prompt templates for the current dataset.
        By default, it returns the ImageNet prompt templates.
        """
        return templates['imagenet']


def _get_mask_unlabeled(train_dataset, setting: ContinualDataset):
    if setting.args.label_perc == 1:
        return np.zeros(train_dataset.targets.shape[0]).astype('bool')
    else:
        lpc = int(setting.args.label_perc * (train_dataset.targets.shape[0] // setting.N_CLASSES_PER_TASK))
        ind = np.indices(train_dataset.targets.shape)[0]
        mask = []
        for i_label, _ in enumerate(np.unique(train_dataset.targets)):
            partial_targets = train_dataset.targets[train_dataset.targets == i_label]
            current_mask = np.random.choice(partial_targets.shape[0], max(
                partial_targets.shape[0] - lpc, 0), replace=False)

            mask = np.append(mask, ind[train_dataset.targets == i_label][current_mask])

        return mask.astype(np.int32)


def _prepare_data_loaders(train_dataset, test_dataset, setting: ContinualDataset):
    if isinstance(train_dataset.targets, list) or not train_dataset.targets.dtype is torch.long:
        train_dataset.targets = torch.tensor(train_dataset.targets, dtype=torch.long)
    if isinstance(test_dataset.targets, list) or not test_dataset.targets.dtype is torch.long:
        test_dataset.targets = torch.tensor(test_dataset.targets, dtype=torch.long)

    setting.unlabeled_mask = _get_mask_unlabeled(train_dataset, setting)

    if setting.unlabeled_mask.sum() != 0:
        train_dataset.targets[setting.unlabeled_mask] = -1  # -1 is the unlabeled class

    return train_dataset, test_dataset


def store_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.

    Attributes:
        train_dataset (Dataset): the training dataset
        test_dataset (Dataset): the test dataset
        setting (ContinualDataset): the setting of the dataset

    Returns:
        the training and test loaders
    """
    setting.args.permute_classes = False
    
    if setting.SETTING == 'task-il' or setting.SETTING == 'class-il':
        setting.c_task += 1

    if not isinstance(train_dataset.targets, np.ndarray):
        train_dataset.targets = np.array(train_dataset.targets)
    if not isinstance(test_dataset.targets, np.ndarray):
        test_dataset.targets = np.array(test_dataset.targets)
    
    if setting.args.permute_classes:
        train_dataset.targets = setting.args.class_order[train_dataset.targets]
        test_dataset.targets = setting.args.class_order[test_dataset.targets]

    # # 添加打印来检查重排序后的标签分布
    # print("Reordered training labels statistics:")
    # unique_labels, counts = np.unique(train_dataset.targets, return_counts=True)
    # for label, count in zip(unique_labels, counts):
    #     print(f"Label {label}: {count} samples")
    
    if setting.args.validation:
        train_idxs, val_idxs = get_validation_indexes(setting.args.validation, train_dataset, setting.args.seed)

        test_dataset.data = train_dataset.data[val_idxs]
        test_dataset.targets = train_dataset.targets[val_idxs]

        train_dataset.data = train_dataset.data[train_idxs]
        train_dataset.targets = train_dataset.targets[train_idxs]

    start_c, end_c = setting.get_offsets()
    
    if setting.SETTING == 'class-il' or setting.SETTING == 'task-il':
        train_mask = np.logical_and(train_dataset.targets >= start_c,
                                    train_dataset.targets < end_c)
        
        # # 统计选出的标签数据
        # unique_labels, counts = np.unique(train_dataset.targets[train_mask], return_counts=True)
        # print("Selected training labels statistics:")
        # for label, count in zip(unique_labels, counts):
        #     print(f"Label {label}: {count} samples")
        
        if setting.args.validation_mode == 'current':
            test_mask = np.logical_and(test_dataset.targets >= start_c,
                                       test_dataset.targets < end_c)
        elif setting.args.validation_mode == 'complete':
            test_mask = np.logical_and(test_dataset.targets >= 0,
                                       test_dataset.targets < end_c)
        else:
            raise ValueError('Unknown validation mode: {}'.format(setting.args.validation_mode))

        test_dataset.data = test_dataset.data[test_mask]
        test_dataset.targets = test_dataset.targets[test_mask]

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = train_dataset.targets[train_mask]

    train_dataset, test_dataset = _prepare_data_loaders(train_dataset, test_dataset, setting)

    train_loader = create_seeded_dataloader(setting.args, train_dataset,
                                            batch_size=setting.args.batch_size, shuffle=True)
    test_loader = create_seeded_dataloader(setting.args, test_dataset,
                                           batch_size=setting.args.batch_size, shuffle=False)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    return train_loader, test_loader


def fix_class_names_order(class_names: List[str], args: Namespace) -> List[str]:
    """
    Permutes the order of the class names according to the class order specified in the arguments.
    The order reflects that of `store_masked_loaders`.

    Args:
        class_names: the list of class names. This should contain all classes in the dataset (not just the current task's ones).
        args: the command line arguments

    Returns:
        List[str]: the class names in the correct order
    """
    if args.permute_classes:
        class_names = [class_names[np.where(args.class_order == i)[0][0]] for i in range(len(class_names))]
    return class_names
