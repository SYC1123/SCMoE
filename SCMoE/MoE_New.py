import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from model.unet_parts import *  # 包含 DoubleConv, Down, Up, OutConv 等

# ------------------ MixStyle Module ------------------
class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.alpha = max(1e-5, alpha)  # 确保 alpha 大于零

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B, C, H, W = x.size()
        mu = x.mean(dim=[2,3], keepdim=True)
        sigma = x.std(dim=[2,3], keepdim=True)
        perm = torch.randperm(B, device=x.device)
        mu2, sigma2 = mu[perm], sigma[perm]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B,1,1,1)).to(x.device)
        mu_mix = mu * (1 - lam) + mu2 * lam
        sigma_mix = sigma * (1 - lam) + sigma2 * lam
        return (x - mu) / sigma * sigma_mix + mu_mix

# ------------------ Gradient Reversal Layer ------------------
class GRL(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

# ------------------ Domain Discriminator ------------------
class DomainDiscriminator(nn.Module):
    def __init__(self, in_features, hidden=256, num_domains=5, grl_lambda=1.0):
        super().__init__()
        self.grl_lambda = grl_lambda
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_domains)
        )
    def forward(self, x):
        x = GRL.apply(x, self.grl_lambda)
        # print(f"GRL output shape: {x.shape}")
        # print(f"self.classifier(x) shape: {self.classifier(x).shape}")
        return self.classifier(x)

# ------------------ Domain-Adversarial Gating Network ------------------
class DomainAdversarialGatingNetwork(nn.Module):
    def __init__(self, input_channels, num_experts, num_domains=5):
        super().__init__()
        # U-Net 门控主干
        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.mix1 = MixStyle(p=0.5, alpha=0.1)
        self.down2 = Down(128, 256)
        self.mix2 = MixStyle(p=0.5, alpha=0.1)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        # 域判别器，作用于最深层特征
        self.dom_disc = DomainDiscriminator(in_features=512, hidden=256, num_domains=num_domains)
        # 上采样与输出
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, num_experts)

    def forward(self, x, domain_id=None):
        # 下采样 + MixStyle
        x1 = self.inc(x)
        x2 = self.down1(x1); x2 = self.mix1(x2)
        x3 = self.down2(x2); x3 = self.mix2(x3)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 对抗分支：域判别损失
        feat = F.adaptive_avg_pool2d(x5, (1,1)).flatten(1)
        # print(f"feat shape: {feat.shape}")
        dom_logits = self.dom_disc(feat)
        # print(f"dom_logits shape: {dom_logits.shape}")
        dom_loss = None
        if self.training and domain_id is not None:
            # 检查 domain_id 的值范围
            if not torch.all(domain_id >= 0) or not torch.all(domain_id < 5):
                raise ValueError(f"domain_id contain values outside [0, 4]: {domain_id.min()}, {domain_id.max()}")

            # 检查 dom_logits 的维度和值范围
            # if len(dom_logits.shape) != 2 or dom_logits.shape[1] != 5:
            #     raise ValueError(f"dom_logits shape is incorrect: {dom_logits.shape}")
            # if not torch.all(dom_logits >= 0) or not torch.all(dom_logits <= 1):
            #     raise ValueError(f"dom_logits contain values outside [0, 1]: {dom_logits.min()}, {dom_logits.max()}")

            dom_loss = F.cross_entropy(dom_logits, domain_id)

        # 上采样解码 + Gate logits
        u = self.up1(x5, x4)
        u = self.up2(u, x3)
        u = self.up3(u, x2)
        u = self.up4(u, x1)
        logits = self.outc(u)
        gate = torch.softmax(logits, dim=1)
        return gate, dom_loss

# ------------------ Uncertainty-Aware Fusion ------------------
class UncertaintyAwareFusion(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp

    def forward(self, expert_outputs, gate_weights):
        # 确保 expert_outputs 的值在 [0, 1] 范围内
        expert_outputs = torch.clamp(expert_outputs, min=1e-5, max=1-1e-5)
        eps = 1e-12

        # 计算熵
        entropy = -expert_outputs * torch.log(expert_outputs + eps) - (1 - expert_outputs) * torch.log(1 - expert_outputs + eps)
        entropy = entropy.mean(dim=2)  # 沿通道维度计算平均熵

        # 缩放熵
        scaled = torch.tanh(entropy / self.temp)
        conf = 1 - scaled

        # 计算校准后的权重
        cal_w = gate_weights * conf
        cal_w = cal_w / (cal_w.sum(dim=1, keepdim=True) + 1e-8)

        return cal_w
# ------------------ Spatio-Temporal Router ------------------
class SpatioTemporalRouter(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        mid = max(1, num_experts//4)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_experts, mid, 1), nn.ReLU(),
            nn.Conv2d(mid, num_experts,1), nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(num_experts,1,3,padding=1), nn.Sigmoid()
        )
    def forward(self, w):
        ca = self.channel_att(w)
        sa = self.spatial_att(w)
        out = w * ca * sa
        return out / (out.sum(dim=1, keepdim=True)+1e-8)

# ------------------ Mixture-of-Experts ------------------
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, input_shape, n_classes, trained_experts, num_domains=5, temp=0.5):
        super().__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.gating = DomainAdversarialGatingNetwork(input_shape[0], num_experts, num_domains)
        self.uncertainty = UncertaintyAwareFusion(temp)
        self.router = SpatioTemporalRouter(num_experts)

    def forward(self, x, domain_id=None):
        # 专家输出
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        # 确保 expert_outs 的值在 [0, 1] 范围内
        #expert_outs = torch.clamp(expert_outs, min=1e-5, max=1-1e-5)

        # 门控 + 对抗损失
        gate, dom_loss = self.gating(x, domain_id)
        w = gate
        # 熵加权融合
        w = self.uncertainty(expert_outs, gate)

        # 时空注意
        w = self.router(w)
        # 检查 w 和 expert_outs 的值范围
        # if not torch.all(w >= 0) or not torch.all(w <= 1):
        #     raise ValueError(f"w contain values outside [0, 1]: {w.min()}, {w.max()}")
        # if not torch.all(expert_outs >= 0) or not torch.all(expert_outs <= 1):
        #     raise ValueError(f"expert_outs contain values outside [0, 1]: {expert_outs.min()}, {expert_outs.max()}")
        #print(123)
        # 加权求和
        out = torch.sum(w.unsqueeze(2) * expert_outs, dim=1)
        #print(456)
        return out, dom_loss