import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import copy
from collections import Counter
from tqdm import tqdm  # 进度条
from joblib import Parallel, delayed
from geomloss import SamplesLoss  # Wasserstein 距离

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from backbone.vit import VisionTransformer
from backbone.resnet34 import BirdResnet

class ExpertLayer(nn.Module):
    def __init__(self, input_dim=768*2, adapt_dim=768, num_classes=400):
        super(ExpertLayer, self).__init__()
        self.adaptive_layer = nn.Linear(input_dim, adapt_dim)
        self.classifier = nn.Linear(adapt_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.adaptive_layer(x)
        x = self.relu(x)
        output = self.classifier(x)
        return output

class MultiLayerFeatureFusion(nn.Module):
    def __init__(self, feature_dim=768, num_layers=3, fuse_mode='attention', hidden_dim=256, dropout=0.1):
        super().__init__()
        self.fuse_mode = fuse_mode
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.temperature = 0.5

        if fuse_mode == 'concat':
            self.proj = nn.Sequential(
                nn.Linear(num_layers * feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim)
            )
        elif fuse_mode == 'attention':
            self.attn = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"Unsupported fuse_mode: {fuse_mode}")

    def forward(self, features):
        if self.fuse_mode == 'concat':
            x = torch.cat(features, dim=-1)  # [B, D * L]
            return self.proj(x)              # [B, D]
        elif self.fuse_mode == 'attention':
            x = torch.stack(features, dim=1)  # [B, L, D]
            attn_scores = self.attn(x).squeeze(-1)  # [B, L]
            attn_weights = F.softmax(attn_scores / self.temperature, dim=-1)  # [B, L]
            fused = (attn_weights.unsqueeze(-1) * x).sum(dim=1)  # [B, D]
            return fused

def wasserstein_distance(feats1, feats2):
    std = max(feats1.std().item(), 1e-3)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=min(0.5, std))
    wasserstein_loss = loss(feats1, feats2)

    return wasserstein_loss

class FusionDistiller(nn.Module):
    def __init__(self, feature_dim=768, num_layers=3, fuse_mode='attention',
                 hidden_dim=256, dropout=0.1, loss_fn='mse'):
        super().__init__()
        self.fuser = MultiLayerFeatureFusion(
            feature_dim=feature_dim,
            num_layers=num_layers,
            fuse_mode=fuse_mode,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        self.loss_fn = loss_fn

    def forward(self, student_feats, teacher_feats):
        s = self.fuser(student_feats)                  # [B, D]
        t = self.fuser([f.detach() for f in teacher_feats])  # [B, D]

        if self.loss_fn == 'mse':
            return F.mse_loss(s, t)
        elif self.loss_fn == 'cosine':
            return 1 - F.cosine_similarity(s, t, dim=-1).mean()
        elif self.loss_fn == 'w_distance':
            return wasserstein_distance(s, t).mean()
        else:
            raise ValueError(f"Unsupported loss_fn: {self.loss_fn}")


class My_FineTune(ContinualModel):
    NAME = 'kdft0401-Fusedfeats-amend'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via Fine-tuning Vision Transformer.')
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net1 = VisionTransformer(num_classes=200)
        self.net1.load_state_dict(torch.load('vit_model_weights_in21k_ft_in1k.pth'))
        self.net1 = self.net1.to(self.device)

        self.net2 = VisionTransformer(num_classes=200)
        self.net2.load_state_dict(torch.load('vit_model_weights_in21k.pth'))
        self.net2 = self.net2.to(self.device)

        for param in self.net1.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False

        self.expert_list = None
        self.n_epochs = args.n_epochs
        self.net1_clone = None
        self.net2_clone = None
        self.atten_net1 = None
        self.atten_net2 = None

        self.opt = None
        self.opt_sched = None
        self.net1_opt = None
        self.net1_opt_sched = None
        self.net2_opt = None
        self.net2_opt_sched = None
        
        self.sel_opt1 = None
        self.sel_opt1_sched = None
        self.sel_opt2 = None
        self.sel_opt2_sched = None

        self.current_task_num = None
        self.count = 0
        self.scale = 1.0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.current_task_num == 0:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            self.net1_opt.zero_grad()
            self.net2_opt.zero_grad()
            
            feats1 = self.net1(inputs, returnt='features')
            feats2 = self.net2(inputs, returnt='features')
            feats = torch.cat((feats1, feats2), dim=1)
            outputs = self.expert_list[self.current_task_num](feats)
            
            tot_loss = self.loss(outputs, labels)
            tot_loss.backward()
            
            self.opt.step()
            self.net1_opt.step()
            self.net2_opt.step()
            
            self.opt_sched.step()
            self.net1_opt_sched.step()
            self.net2_opt_sched.step()
            
            tot_loss = tot_loss.item()
            
        else:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.opt.zero_grad()
            self.net1_opt.zero_grad()
            self.net2_opt.zero_grad()
            self.selct_opt1.zero_grad()
            self.selct_opt2.zero_grad()
            
            # student forward
            feats1, (feats1_1, feats1_2, feats1_3) = self.net1(inputs, returnt='both')
            feats2, (feats2_1, feats2_2, feats2_3) = self.net2(inputs, returnt='both')
            
            # teacher forward
            feats1_clone, (feats1_1_clone, feats1_2_clone, feats1_3_clone) = self.net1_clone(inputs, returnt='both')
            feats2_clone, (feats2_1_clone, feats2_2_clone, feats2_3_clone) = self.net2_clone(inputs, returnt='both')
        
            student1_feats = [feats1_1, feats1_2, feats1_3]
            student2_feats = [feats2_1, feats2_2, feats2_3]
            teacher1_feats = [feats1_1_clone, feats1_2_clone, feats1_3_clone]
            teacher2_feats = [feats2_1_clone, feats2_2_clone, feats2_3_clone]
            feats = torch.cat((feats1_1, feats2_1), dim=1)
            feats_clone = torch.cat((feats1_1_clone, feats2_1_clone), dim=1)
            
            outputs = self.expert_list[self.current_task_num](feats)
            tot_loss = self.loss(outputs, labels)
            
            loss_train = tot_loss.item()

            for i in range(self.current_task_num):
                out1 = self.expert_list[i](feats_clone)
                out2 = self.expert_list[i](feats)
                tot_loss += self.KD_loss(out1, out2)
            
            loss_kd = tot_loss.item() - loss_train
            
            # 融合后的蒸馏损失（基于w-distance）
            loss_w2_net1 = self.selector1(student1_feats, teacher1_feats)
            loss_w2_net2 = self.selector2(student2_feats, teacher2_feats)
            loss_w2 = loss_w2_net1 + loss_w2_net2
            tot_loss = tot_loss + loss_w2
            tot_loss.backward()
        
            # 打印信息
            # if self.count % 20 == 0:
            #     print(f"\n训练损失：{loss_train:.5f}  KD损失：{loss_kd:.5f}  loss_w2：{loss_w2.item():.5f}")
            #     print(f"loss_w2_net1: {loss_w2_net1.item():.5f}, loss_w2_net2: {loss_w2_net2.item():.5f}")
            #     print("------------------------------------------")

            torch.nn.utils.clip_grad_norm_(self.net1.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(self.net2.parameters(), max_norm=5.0)
                    
            self.opt.step()
            self.net1_opt.step()
            self.net2_opt.step()
            self.selct_opt1.step()
            self.selct_opt2.step()
            
            self.opt_sched.step()
            self.net1_opt_sched.step()
            self.net2_opt_sched.step()
            self.selct_opt1_sched.step()
            self.selct_opt2_sched.step()

            tot_loss = tot_loss.item()

        self.count += 1
        return tot_loss

    def W_distance(self, feats1, feats2, weight=1.0, norm_method="None"):

        feats1 = self.normalize_features(feats1, method=norm_method)
        feats2 = self.normalize_features(feats2, method=norm_method)
        
        std = max(feats1.std().item(), 1e-3)
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=min(0.5, std))
    
        wasserstein_loss = loss(feats1, feats2)
    
        return weight * wasserstein_loss

    def normalize_features(self, features, method="L2"):
        device = features.device
        
        if method == "L2":
            return F.normalize(features, p=2, dim=1)
        
        elif method == "BN":
            bn_layer = torch.nn.BatchNorm1d(features.shape[1], affine=False).to(device)
            return bn_layer(features.to(device))
    
        elif method == "None":
            return features.to(device)
    
        elif method == "MinMax":
            min_val, max_val = features.min(dim=1, keepdim=True)[0].to(device), features.max(dim=1, keepdim=True)[0].to(device)
            return (features - min_val) / (max_val - min_val + 1e-6)
    
        elif method == "STD":
            mean, std = features.mean(dim=1, keepdim=True).to(device), features.std(dim=1, keepdim=True).to(device) + 1e-6
            return (features - mean) / std
    
        elif method == "Global-Zscore":
            mean, std = features.mean().to(device), features.std().to(device) + 1e-6
            return (features - mean) / std
    
        elif method == "Log":
            return torch.log1p(features).to(device) 
    
        elif method == "PCA-Whiten":
            mean = features.mean(dim=0, keepdim=True).to(device)
            features_centered = features - mean
            cov_matrix = torch.mm(features_centered.T, features_centered) / (features.shape[0] - 1)
            cov_matrix = cov_matrix.to(device)
            U, S, V = torch.svd(cov_matrix)
            whitening_matrix = torch.mm(U, torch.diag(1.0 / (S + 1e-6))).to(device)
            return torch.mm(features_centered, whitening_matrix)
    
        else:
            raise ValueError(f"未知归一化方法: {method}")
    
    def KD_loss(self, out1, out2, feats_t=None, feats_s=None, T=5.0, mode="KL"):
        if mode == "KL":
            loss = F.kl_div(
                F.log_softmax(out2 / T, dim=1),  # Student softmax
                F.softmax(out1 / T, dim=1),  # Teacher softmax
                reduction='batchmean'
            ) * (T * T) 
            return loss
    
        elif mode == "MSE":
            soft_target = F.softmax(out1 / T, dim=1)
            soft_prediction = F.softmax(out2 / T, dim=1)
            return F.mse_loss(soft_prediction, soft_target)
    
        elif mode == "FeatureMatching":
            if feats_t is None or feats_s is None:
                raise ValueError("FeatureMatching 需要提供 feats_t (Teacher) 和 feats_s (Student) 特征!")
            return F.mse_loss(feats_s, feats_t)
    
        elif mode == "Wasserstein":
            if feats_t is None or feats_s is None:
                raise ValueError("Wasserstein 需要提供 feats_t (Teacher) 和 feats_s (Student) 特征!")
            
            feats_t = self.normalize_L2(feats_t)
            feats_s = self.normalize_L2(feats_s)
    
            blur_value = max(0.05, min(0.2, feats_t.std().item()))  # 避免数值不稳定
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=blur_value)
    
            wasserstein_loss = loss(feats_t, feats_s)
            return torch.tanh(wasserstein_loss / 10.0) * 10.0  # Soft Clipping
    
        else:
            raise ValueError(f"未知的蒸馏损失模式: {mode}，可选值: KL, MSE, FeatureMatching, Wasserstein")


    def unfreeze_backbones(self, train_loader):
        assert isinstance(self.net1, VisionTransformer), "self.net1 不是 VisionTransformer"
        assert isinstance(self.net2, VisionTransformer), "self.net2 不是 VisionTransformer"
    
        total_layers = len(self.net1.blocks)
        assert total_layers >= 3, "ViT 层数不足，无法解冻最后 3 层"
    
        for layer in range(total_layers - 3, total_layers):
            for param in self.net1.blocks[layer].parameters():
                param.requires_grad = True
    
        for layer in range(total_layers - 3, total_layers):
            for param in self.net2.blocks[layer].parameters():
                param.requires_grad = True
    
        for param in self.net1.norm.parameters():
            param.requires_grad = True
        for param in self.net2.norm.parameters():
            param.requires_grad = True

        self.net1_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net1.parameters()), lr=0.0001, weight_decay=1e-5)
        self.net2_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net2.parameters()), lr=0.0001, weight_decay=1e-5)
        self.net1_opt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.net1_opt, T_max=self.n_epochs, eta_min=1e-5)
        self.net2_opt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.net2_opt, T_max=self.n_epochs, eta_min=1e-5)
        print("net1, net2已解冻后三层，优化器已创建")
	
    def new_expert_list(self, n_classes, length):
        self.expert_list = [ExpertLayer(num_classes=n_classes).to(self.device) for _ in range(length)]
        print(f"专家列表已创建，长度{len(self.expert_list)}，输出头维度{n_classes}")
    
    def update_expert(self, train_loader, idx):
        for expert in self.expert_list:
            for param in expert.parameters():
                param.requires_grad = False
        print("所有expert训练参数已冻结")
        
        for param in self.expert_list[idx].parameters():
            param.requires_grad = True
        print(f"expert{idx}训练参数已解冻")
        
        params_to_optimize = list(self.expert_list[idx].parameters())
        self.opt = torch.optim.Adam(params_to_optimize, lr=0.0005)
        self.opt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=len(train_loader) * self.n_epochs, eta_min=1e-5)
        print(f"expert{idx}已存入优化器当前学习率：", self.opt.param_groups[0]["lr"])
    
    def new_selector(self, train_loader):
        self.selector1 = FusionDistiller(feature_dim=768, num_layers=3, loss_fn='w_distance').to(self.device)
        self.selector2 = FusionDistiller(feature_dim=768, num_layers=3, loss_fn='w_distance').to(self.device)

        self.selct_opt1 = torch.optim.AdamW(self.selector1.parameters(), lr=0.0005, weight_decay=1e-3)
        self.selct_opt1_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.selct_opt1, T_max=self.n_epochs * len(train_loader), eta_min=1e-5)
    
        self.selct_opt2 = torch.optim.AdamW(self.selector2.parameters(), lr=0.0005, weight_decay=1e-3)
        self.selct_opt2_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.selct_opt2, T_max=self.n_epochs * len(train_loader), eta_min=1e-5)
        print("新的 selector 已创建（temperature↑, entropy_reg↑, weight_decay↑）")
   
    def reset_opt_sched(self, train_loader):
        self.net1_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net1.parameters()), lr=0.0001, weight_decay=1e-5)
        self.net2_opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net2.parameters()), lr=0.0001, weight_decay=1e-5)
        self.net1_opt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.net1_opt, T_max=self.n_epochs, eta_min=1e-5)
        self.net2_opt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.net2_opt, T_max=self.n_epochs, eta_min=1e-5)

    def clone_backbones(self):
        self.net1_clone, self.net2_clone = copy.deepcopy(self.net1), copy.deepcopy(self.net2)
        self.net1_clone.eval(), self.net2_clone.eval()
        for param in self.net1_clone.parameters():
            param.requires_grad = False
        for param in self.net2_clone.parameters():
            param.requires_grad = False
        print("net1, net2已克隆冻结备份")






