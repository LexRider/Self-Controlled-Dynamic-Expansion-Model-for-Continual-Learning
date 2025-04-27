import torch.nn as nn

def get_trainable_parameters(model):
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    print(f"{'Parameter Name':<60} {'Trainable':<10} {'Shape':<25} {'#Params'}")
    print("=" * 110)

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        if param.requires_grad:
            status = "✅ Yes"
            trainable_params += numel
        else:
            status = "❌ No"
            frozen_params += numel

        print(f"{name:<60} {status:<10} {str(tuple(param.shape)):<25} {numel:,}")

    print("=" * 110)
    print(f"🔢 Total Parameters      : {total_params:,}")
    print(f"✅ Trainable Parameters : {trainable_params:,}")
    print(f"❌ Frozen Parameters    : {frozen_params:,}")

def ablation_unfreeze_backbones(model, name=None):
    import torch
    num_unfrozen = 0

    if name == 'vit':
        print("==> 解冻 VisionTransformer 的后 3 层和所有 LayerNorm")

        # 解冻最后3个 transformer block
        for block in model.transformer.resblocks[-3:]:
            for param in block.parameters():
                param.requires_grad = True
                num_unfrozen += param.numel()

        # 解冻 ln_post
        model.ln_post.weight.requires_grad = True
        model.ln_post.bias.requires_grad = True
        num_unfrozen += model.ln_post.weight.numel() + model.ln_post.bias.numel()

        # 解冻 proj
        model.proj.requires_grad = True
        num_unfrozen += model.proj.numel()

    elif name == 'resnet':
        print("==> 解冻 ModifiedResNet 的最后一层和 attnpool")

        for param in model.layer4.parameters():
            param.requires_grad = True
            num_unfrozen += param.numel()

        for param in model.attnpool.parameters():
            param.requires_grad = True
            num_unfrozen += param.numel()

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # 🔄 将所有参数转成 float32
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            param.data = param.data.float()  # inplace 修改参数值为 float32
            if param._grad is not None:
                param._grad = param._grad.float()  # 若已有梯度，也转换

    print(f"✅ 解冻完成，共解冻参数数量：{num_unfrozen:,} 个，全部转为 float32 类型")



# def ablation_unfreeze_backbones(model, name = None):
#     # 解冻参数计数器（可选）
#     num_unfrozen = 0

#     # Vision Transformer 的判断标准
#     if name == 'vit':
#         print("==> 解冻 VisionTransformer 的后 3 层和所有 LayerNorm")
#         # 解冻最后3个 transformer block
#         for block in model.transformer.resblocks[-3:]:
#             for param in block.parameters():
#                 param.requires_grad = True
#                 num_unfrozen += param.numel()

#         # 解冻 ln_post
#         model.ln_post.weight.requires_grad = True
#         model.ln_post.bias.requires_grad = True
#         num_unfrozen += model.ln_post.weight.numel() + model.ln_post.bias.numel()

#         # 解冻 proj
#         model.proj.requires_grad = True
#         num_unfrozen += model.proj.numel()

#     # Modified ResNet 的判断标准
#     elif name == 'resnet':
#         print("==> 解冻 ModifiedResNet 的最后一层和 attnpool")
#         # 解冻最后一层 layer4
#         for param in model.layer4.parameters():
#             param.requires_grad = True
#             num_unfrozen += param.numel()

#         # 解冻 attnpool
#         for param in model.attnpool.parameters():
#             param.requires_grad = True
#             num_unfrozen += param.numel()

#     else:
#         raise ValueError(f"Unsupported model type: {type(model)}")

#     print(f"✅ 解冻完成，共解冻参数数量：{num_unfrozen:,} 个")