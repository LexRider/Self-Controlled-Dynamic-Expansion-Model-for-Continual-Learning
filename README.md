# Self-Controlled Dynamic Expansion Model for Continual Learning

Official implementation of **Self-Controlled Dynamic Expansion Model for Continual Learning (SCDEM)**.

[Paper](https://arxiv.org/abs/2504.10561)

## Overview

SCDEM addresses continual learning with multiple trainable pretrained Vision Transformer backbones. It expands a lightweight expert for each new task while collaboratively adapting shared representations without erasing previously acquired knowledge.

The method combines:

- **task-wise dynamic expert expansion**;
- **collaborative backbone optimization** using historical expert predictions;
- **feature distribution consistency** based on optimal-transport distances;
- **dynamic layer-wise feature attention** for adaptive regularization across representation layers.

## Main components

- **Multi-backbone representation learning:** two differently pretrained ViT backbones provide complementary features.
- **Lightweight task experts:** a compact classifier is assigned to each task.
- **Historical-expert distillation:** previous experts constrain representation updates during later tasks.
- **Feature Distribution Consistency (FDC):** Wasserstein/Sinkhorn losses align old and new feature distributions.
- **Dynamic Layer-Wise Feature Attention:** learned fusion weights determine the contribution of different ViT layers to the consistency objective.
- **Selective fine-tuning:** only the final ViT blocks and normalization layers are unfrozen in the current implementation.

## Repository structure

```text
.
├── backbone/              # Vision Transformer and auxiliary backbones
├── datasets/              # Standard and cross-domain task sequences
├── models/
│   └── SCDEM.py           # Main SCDEM implementation
├── utils/
│   ├── main.py            # Training entry point
│   ├── training.py        # SCDEM-aware training and evaluation loop
│   └── ...                # Logging, checkpoints, metrics, and utilities
└── scripts/               # Local and Slurm launch helpers
```

## Installation

Create an isolated Python environment and install a PyTorch build compatible with your hardware first. The implementation additionally uses:

```bash
pip install torchvision timm kornia geomloss numpy pandas matplotlib tqdm \
    pyyaml wandb torchinfo deeplake requests joblib python-dotenv GitPython \
    setproctitle six onedrivedownloader googledrivedownloader
```

Some optional datasets may require extra packages.

## Required pretrained checkpoints

Place the following compatible Vision Transformer checkpoints in the repository root:

```text
vit_model_weights_in21k_ft_in1k.pth
vit_model_weights_in21k.pth
```

They are loaded directly by `models/SCDEM.py`. Adjust the filenames or loading code when using another checkpoint layout. Checkpoint acquisition and use should follow the licenses of the corresponding pretrained models.

## Before running this code snapshot

The public filename and the internal model identifier are not aligned in the uploaded snapshot. In `models/SCDEM.py`, change:

```python
NAME = 'kdft0401-Fusedfeats-amend'
```

to:

```python
NAME = 'SCDEM'
```

The model parser inherits a rehearsal-style `--buffer_size` argument from the framework, although the current SCDEM implementation does not use an exemplar replay buffer as its main mechanism.

## Data

Representative dataset identifiers included in this repository are:

- `seq-crossdomain`
- `seq-tinycub`
- `seq-tinybird`
- `seq-tinycifar100birds`
- `seq-cifar10birds`
- `seq-cifar100tiny`
- `seq-cifar100224`
- `seq-tiny224`
- `seq-chestx`
- `seq-isic`
- `seq-eurosat-rgb`
- `seq-resisc45`
- `seq-imagenet-r`

Datasets and results are stored below `./data/` by default. Use `--base_path` to select another location. Refer to the corresponding loader in `datasets/` for manually prepared directory structures.

## Quick start

After aligning the model identifier and placing the pretrained checkpoints, a typical command is:

```bash
python utils/main.py \
    --model SCDEM \
    --dataset seq-crossdomain \
    --lr 0.0005 \
    --batch_size 32 \
    --n_epochs 5 \
    --buffer_size 200 \
    --seed 0
```

This is a usage template rather than an exact reproduction command. Match the dataset sequence, task split, trainable-layer setting, number of epochs, and optimizer configuration to the target experiment.

## Implementation notes

- `models/SCDEM.py` constructs two ViT backbones and one expert per task.
- The current `unfreeze_backbones` implementation unfreezes the final three transformer blocks and final normalization layer of each backbone.
- The expert optimizer, backbone optimizers, selector optimizers, and cosine schedules use learning rates defined directly in the model code.
- Historical backbone copies are frozen and used as teachers for later tasks.
- Optimal-transport consistency is implemented with `geomloss.SamplesLoss`.

## Logging and reproducibility

- Set `--seed` to control random initialization and class ordering.
- Use `--debug_mode 1` for a short smoke test.
- Checkpoints can be saved with `--savecheck last` or `--savecheck task`.
- Weights & Biases logging is disabled when project/account arguments are not provided.

## Citation

```bibtex
@article{wu2025scdem,
  title   = {Self-Controlled Dynamic Expansion Model for Continual Learning},
  author  = {Wu, Runqing and Huang, Kaihui and Zhang, Hanyi and Ye, Fei},
  journal = {arXiv preprint arXiv:2504.10561},
  year    = {2025}
}
```

## Acknowledgements

The code follows a Mammoth-style continual-learning framework and uses pretrained Vision Transformer representations. Please cite the relevant framework and backbone works when building on this repository.
