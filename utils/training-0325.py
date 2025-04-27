# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# training 0315

from copy import deepcopy
import math
import os
import sys
from argparse import Namespace
from time import time
from typing import Iterable, Tuple

import torch
from tqdm import tqdm
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel

from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import *
from utils.stats import track_system_stats
from torch.nn import functional as F
from collections import defaultdict
from typing import Dict, Iterable
from time import time

try:
    import wandb
except ImportError:
    wandb = None

from collections import Counter
from torch.utils.data import DataLoader

def print_label_distribution(train_loader: Iterable) -> Dict[int, int]:
    """
    Computes and prints the distribution of labels in the training data loader.

    Args:
        train_loader: The data loader for the training set.

    Returns:
        A dictionary where keys are the unique labels and values are their corresponding counts.
    """
    label_count = defaultdict(int)

    for data in train_loader:
        labels = data[1]

        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()

        for label in labels:
            label_count[label] += 1

    label_count = dict(label_count)

    print("Label Distribution in Training Set:")
    for label, count in sorted(label_count.items()):
        print(f"Label {label}: {count} samples")

    return label_count

def count_params(optimizer):
    """
    计算并打印优化器中可训练和不可训练参数的数量。

    参数:
    optimizer: PyTorch优化器对象
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    # 遍历优化器中的所有参数
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
            else:
                non_trainable_params += param_count

    # 打印结果
    print(f"总参数数量: {total_params}")
    print(f"可训练参数数量: {trainable_params}")
    print(f"不可训练参数数量: {non_trainable_params}")

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    num_classes = dataset.N_CLASSES
    start_c, end_c = dataset.get_offsets(k)
    outputs[:, :start_c] = -float('inf')
    outputs[:, end_c:num_classes] = -float('inf')

def entropy(p):
    return -(p * torch.log(p + 1e-8)).sum(dim=1)

@torch.no_grad()
def class_il_predict(feats, expert_list, mode='avg', fixed_weights=None,
                     entropy_gamma=1.0, topk=2, verbose=True):

    device = feats.device
    num_experts = len(expert_list)
    B = feats.size(0)

    softmax_outputs = []
    entropy_list = []

    for idx, expert in enumerate(expert_list):
        logits = expert(feats)  # [B, C]
        probs = F.softmax(logits, dim=1)
        softmax_outputs.append(probs)
        ent = entropy(probs)  # [B]
        entropy_list.append(ent)

        if verbose and mode in ['entropy', 'topk_entropy', 'min_entropy']:
            ent_mean = ent.mean().item()
            print(f"[Expert {idx}] 平均熵: {ent_mean:.4f}")

    # [K, B, C], [K, B]
    stacked_outputs = torch.stack(softmax_outputs, dim=0)
    entropies = torch.stack(entropy_list, dim=0)

    if mode == 'avg':
        outputs = stacked_outputs.mean(dim=0)

    elif mode == 'entropy':
        weights = torch.exp(entropy_gamma / (entropies + 1e-6))  # [K, B]
        norm_weights = weights / weights.sum(dim=0, keepdim=True)  # [K, B]
        outputs = (norm_weights.unsqueeze(-1) * stacked_outputs).sum(dim=0)

    elif mode == 'min_entropy':
        min_indices = torch.argmin(entropies, dim=0)  # [B]
        print(min_indices)
        outputs = []
        for b in range(B):
            selected = stacked_outputs[min_indices[b], b]  # [C]
            outputs.append(selected)
        outputs = torch.stack(outputs, dim=0)

    else:
        raise ValueError(f"不支持的 mode：{mode}")

    return outputs

@torch.no_grad()
def select_most_confident_expert(feats, expert_list, verbose=True):
    """
    在多个 expert 中选择平均熵最小的 expert，并返回其输出。

    参数：
    - feats: [B, D] 输入特征
    - expert_list: List[nn.Module] 所有专家模型
    - verbose: 是否打印每个 expert 的平均熵

    返回：
    - best_expert_idx: int，平均熵最小的 expert 的索引
    - best_outputs: [B, C] 该 expert 的 softmax 输出
    - avg_entropies: List[float] 每个 expert 的平均熵
    """
    avg_entropies = []
    outputs_list = []

    for idx, expert in enumerate(expert_list):
        logits = expert(feats)                  # [B, C]
        probs = F.softmax(logits, dim=1)        # [B, C]
        outputs_list.append(probs)

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B]
        avg_entropy = entropy.mean().item()
        avg_entropies.append(avg_entropy)

        # if verbose:
        #     print(f"[Expert {idx}] 平均熵: {avg_entropy:.4f}")

    # 找到平均熵最小的专家
    best_expert_idx = int(torch.tensor(avg_entropies).argmin().item())
    best_outputs = outputs_list[best_expert_idx]  # [B, C]

    return best_expert_idx, best_outputs, avg_entropies

@torch.no_grad()
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, return_loss=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last: a boolean indicating whether to evaluate only the last task
        return_loss: a boolean indicating whether to return the loss in addition to the accuracy

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task. If return_loss is True, the loss is also returned as a third element.
    """
    # status = model.net.training
    # model.net.eval()
    status = model.net1.training
    model.net1.eval()
    model.net2.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    loss_fn = dataset.get_loss()
    avg_loss = 0
    total_len = sum(len(x) for x in dataset.test_loaders) if hasattr(dataset.test_loaders[0], '__len__') else None
    pbar = tqdm(dataset.test_loaders, total=total_len, desc='Evaluating')
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue

        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
                
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                if model.args.eval_future and k >= model.current_task:
                    outputs = model.future_forward(inputs)
                else:
                    #################################################################################
                    feats1 = model.net1(inputs, returnt='features')
                    feats2 = model.net2(inputs, returnt='features')
                    if hasattr(model, 'net3'):
                        feats3 = model.net3(inputs)
                        feats3 = feats3.to(dtype=torch.float32)
                        feats = torch.cat((feats1, feats2, feats3), dim=1)
                    else:
                        feats = torch.cat((feats1, feats2), dim=1)
                    outputs = model.expert_list[k](feats)
                    # idxxx, outputs, _ = select_most_confident_expert(feats, model.expert_list)
                    # print(f"最小熵对应专家idx: {idxxx}")
                    #################################################################################
                    
                    # outputs = model(inputs)

            if return_loss:
                loss = loss_fn(outputs, labels)
                avg_loss += loss.item()

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            #################################################################################
            # print("\nlabels content:",  labels.tolist(), "n_classes: ", n_classes)
            # print("pred content  :",  pred.tolist(), "n_classes: ", n_classes)
            #################################################################################
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1
            pbar.set_postfix({f'acc_task_{k+1}': max(0, correct / total * 100)})
            pbar.set_description(f"Evaluating Task {k+1}")
            pbar.update(1)

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    pbar.close()

    # model.net.train(status)
    model.net1.train(status)
    model.net2.train(status)
    if return_loss:
        return accs, accs_mask_classes, avg_loss / total
    return accs, accs_mask_classes


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = args.conf_jobnum.split('-')[0]
    name = f'{run_name}_{run_id}'
    mode = 'disabled' if os.getenv('MAMMOTH_TEST', '0') == '1' else os.getenv('WANDB_MODE', 'online')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name, mode=mode)
    args.wandb_url = wandb.run.get_url()


def train_single_epoch(model: ContinualModel,
                       train_loader: Iterable,
                       args: Namespace,
                       epoch: int,
                       current_task: int,
                       system_tracker=None,
                       data_len=None,
                       scheduler=None) -> int:
    """
    Trains the model for a single epoch.

    Args:
        model: the model to be trained
        train_loader: the data loader for the training set
        args: the arguments from the command line
        epoch: the current epoch
        current_task: the current task index
        system_tracker: the system tracker to monitor the system stats
        data_len: the length of the training data loader. If None, the progress bar will not show the training percentage
        scheduler: the scheduler for the current epoch

    Returns:
        the number of iterations performed in the current epoch
    """
    train_iter = iter(train_loader)

    i = 0
    previous_time = time()

    pbar = tqdm(train_iter, total=data_len, desc=f"Task {current_task + 1} - Epoch {epoch + 1}")
    while True:
        try:
            data = next(train_iter)
        except StopIteration:
            break
        if args.debug_mode and i > model.get_debug_iters():
            break
        if args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
            break

        if hasattr(train_loader.dataset, 'logits'):
            inputs, labels, not_aug_inputs, logits = data
            inputs = inputs.to(model.device)
            labels = labels.to(model.device, dtype=torch.long)
            not_aug_inputs = not_aug_inputs.to(model.device)
            logits = logits.to(model.device)
            loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
        else:
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
            not_aug_inputs = not_aug_inputs.to(model.device)
            loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)
        assert not math.isnan(loss)

        if args.code_optimization == 0 and 'cuda' in str(args.device):
            torch.cuda.synchronize()
        system_tracker()
        i += 1

        time_diff = time() - previous_time
        previous_time = time()
        ep_h = 3600 / (data_len * time_diff) if data_len else 'N/A'
        pbar.set_postfix({'loss': loss} if ep_h == 'N/A' else {'loss': loss, 'ep/h': ep_h})
        pbar.update()

    if scheduler is not None:
        scheduler.step()


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        initialize_wandb(args)

    if not args.disable_log:
        logger = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)

    # model.net.to(model.device)
    model.net1.to(model.device)
    model.net2.to(model.device)
    torch.cuda.empty_cache()

    with track_system_stats(logger) as system_tracker:
        results, results_mask_classes = [], []

        if args.eval_future:
            results_transf, results_mask_classes_transf = [], []

        if args.start_from is not None:
            for i in range(args.start_from):
                train_loader, _ = dataset.get_data_loaders()
                model.meta_begin_task(dataset)
                model.meta_end_task(dataset)

        if args.loadcheck is not None:
            model, past_res = mammoth_load_checkpoint(args, model)

            if not args.disable_log and past_res is not None:
                (results, results_mask_classes, csvdump) = past_res
                logger.load(csvdump)

            print('Checkpoint Loaded!')

        if args.enable_other_metrics:
            dataset_copy = get_dataset(args)
            for t in range(dataset.N_TASKS):
                # model.net.train()
                model.net1.train()
                model.net2.train()
                _, _ = dataset_copy.get_data_loaders()
            if model.NAME != 'icarl' and model.NAME != 'pnn':
                random_results_class, random_results_task = evaluate(model, dataset_copy)

        print(file=sys.stderr)
        start_task = 0 if args.start_from is None else args.start_from
        end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

        if args.eval_future:
            eval_dataset = get_dataset(args)
            for _ in range(dataset.N_TASKS):
                eval_dataset.get_data_loaders()
                model.change_transform(eval_dataset)
                del eval_dataset.train_loader
        else:
            eval_dataset = dataset

        torch.cuda.empty_cache()
        for t in range(start_task, end_task):
            # model.net.train()
            model.net1.train()
            model.net2.train()
            train_loader, _ = dataset.get_data_loaders()
            model.meta_begin_task(dataset)

            #################################################################################            
            # 告诉模型当前在训练哪个任务
            model.current_task_num = t
            # 创建一个新的专家和选择器
            model.new_expert(train_loader)
            model.new_selector(train_loader)

            if t == 0:
                model.unfreeze_backbones(train_loader)
            
            model.reset_opt_sched(train_loader)
            #################################################################################

            if not args.inference_only and args.n_epochs > 0:
                if t and args.enable_other_metrics:
                    accs = evaluate(model, eval_dataset, last=True)
                    results[t - 1] = results[t - 1] + accs[0]
                    if dataset.SETTING == 'class-il':
                        results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

                scheduler = dataset.get_scheduler(model, args, reload_optim=True) if not hasattr(model, 'scheduler') else model.scheduler

                epoch = 0
                best_ea_metric = None
                best_ea_model = None
                cur_stopping_patience = args.early_stopping_patience               
                while True:
                    data_len = None
                    if not isinstance(dataset, GCLDataset):
                        data_len = len(train_loader)

                    train_single_epoch(model, train_loader, args, current_task=t, epoch=epoch,
                                       system_tracker=system_tracker, data_len=data_len, scheduler=scheduler)

                    epoch += 1
                    if args.fitting_mode == 'epochs' and epoch >= model.args.n_epochs:
                        break
                    elif args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
                        break
                    elif args.fitting_mode == 'early_stopping' and epoch % args.early_stopping_freq == 0 and epoch > 0:
                        epoch_accs, _, epoch_loss = evaluate(model, eval_dataset, return_loss=True, last=True)

                        if args.early_stopping_metric == 'accuracy':
                            ea_metric = np.mean(epoch_accs)  # Higher accuracy is better
                        elif args.early_stopping_metric == 'loss':
                            ea_metric = -epoch_loss  # Lower loss is better
                        else:
                            raise ValueError(f'Unknown early stopping metric {args.early_stopping_metric}')

                        # Higher accuracy is better
                        if best_ea_metric is not None and ea_metric - best_ea_metric < args.early_stopping_epsilon:
                            cur_stopping_patience -= args.early_stopping_freq
                            if cur_stopping_patience <= 0:
                                print(f"\nEarly stopping at epoch {epoch} with metric {abs(ea_metric)}", file=sys.stderr)
                                model.load_state_dict({k: v.to(model.device) for k, v in best_ea_model.items()})
                                break
                            print(f"\nNo improvement at epoch {epoch} (best {abs(best_ea_metric)} | current {abs(ea_metric)}). "
                                  f"Waiting for {cur_stopping_patience} epochs to stop.", file=sys.stderr)
                        else:
                            print(f"\nFound better model with metric {abs(ea_metric)} at epoch {epoch}. "
                                  f"Previous value was {abs(best_ea_metric) if best_ea_metric is not None else 'None'}", file=sys.stderr)
                            best_ea_metric = ea_metric
                            best_ea_model = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                            cur_stopping_patience = args.early_stopping_patience

                    if args.eval_epochs is not None and (epoch > 0 or args.eval_epochs == 1) and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs:
                        epoch_accs = evaluate(model, eval_dataset)

                        log_accs(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)
            #################################################################################

            model.clone_backbones()
            model.count = 0

            #################################################################################
            
            model.meta_end_task(dataset)
            accs = evaluate(model, eval_dataset)

            if args.eval_future and t < dataset.N_TASKS - 1:
                transf_accs = accs[0][t + 1:], accs[1][t + 1:]
                accs = accs[0][:t + 1], accs[1][:t + 1]
                results_transf.append(transf_accs[0])
                results_mask_classes_transf.append(transf_accs[1])

            results.append(accs[0])
            results_mask_classes.append(accs[1])

            log_accs(args, logger, accs, t, dataset.SETTING)

            if args.eval_future:
                avg_transf = np.mean([np.mean(task_) for task_ in results_transf])
                print(f"Transfer Metrics  -  AVG Transfer {avg_transf:.2f}")
                if t < dataset.N_TASKS - 1:
                    log_accs(args, logger, transf_accs, t, dataset.SETTING, future=True)

            if args.savecheck:
                save_obj = {
                    'model': model.state_dict(),
                    'args': args,
                    'results': [results, results_mask_classes, logger.dump()],
                    'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                }
                if 'buffer_size' in model.args:
                    save_obj['buffer'] = deepcopy(model.buffer).to('cpu')

                # Saving model checkpoint for the current task
                checkpoint_name = None
                if args.savecheck == 'task':
                    checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_{t}.pt'
                elif args.savecheck == 'last' and t == end_task - 1:
                    checkpoint_name = f'checkpoints/{args.ckpt_name}_joint.pt' if args.joint else f'checkpoints/{args.ckpt_name}_last.pt'
                if checkpoint_name is not None:
                    torch.save(save_obj, checkpoint_name)

        if args.validation:
            # Final evaluation on the real test set
            print("Starting final evaluation on the real test set...", file=sys.stderr)
            del dataset
            args.validation = None
            args.validation_mode = 'current'

            final_dataset = get_dataset(args)
            for _ in range(final_dataset.N_TASKS):
                final_dataset.get_data_loaders()
            accs = evaluate(model, final_dataset)

            log_accs(args, logger, accs, 'final', final_dataset.SETTING, prefix="FINAL")

        if not args.disable_log and args.enable_other_metrics:
            logger.add_bwt(results, results_mask_classes)
            logger.add_forgetting(results, results_mask_classes)
            if model.NAME != 'icarl' and model.NAME != 'pnn':
                logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
