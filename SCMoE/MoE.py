# import torch
# from torch import nn
# from model.unet_model import UNet
# 定义门控网络
# class MixtureOfExperts(nn.Module):
#     def __init__(self, num_experts, input_shape, n_classes, trained_experts):
#         super(MixtureOfExperts, self).__init__()
#         self.num_experts = num_experts
#         self.experts = nn.ModuleList(trained_experts)
#         self.gating_network = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], 64),
#             nn.ReLU(),
#             nn.Linear(64, num_experts)
#         )
#
#     def forward(self, x):
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # 形状: [batch_size, num_experts, 1, height, width]
#         gating_outputs = torch.softmax(self.gating_network(x), dim=1)  # 形状: [batch_size, num_experts]
#         # 为gating_outputs增加三个维度以匹配expert_outputs
#         gating_outputs = gating_outputs.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # 形状: [batch_size, num_experts, 1, 1, 1]
#         # 通过广播将gating_outputs乘以expert_outputs
#         # 由于expert_outputs有2个通道，我们需要将gating_outputs扩展到匹配这个通道数
#         gating_outputs = gating_outputs.expand(-1, -1, 1, -1, -1)  # 形状: [batch_size, num_experts, 1, 1, 1]
#
#         output = torch.sum(gating_outputs * expert_outputs, dim=1)  # 形状: [batch_size, 1, height, width]
#         return torch.sigmoid(output)  # 直接返回sigmoid激活后的结果
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GatingNetwork(nn.Module):
#     def __init__(self, input_shape, num_experts):
#         super(GatingNetwork, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, num_experts, kernel_size=2, stride=2)
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return torch.softmax(decoded, dim=1)

# class MixtureOfExperts(nn.Module):
#     def __init__(self, num_experts, input_shape, n_classes, trained_experts):
#         super(MixtureOfExperts, self).__init__()
#         self.num_experts = num_experts
#         self.experts = nn.ModuleList(trained_experts)
#         self.gating_network = GatingNetwork(input_shape, num_experts)

#     def forward(self, x):
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # 形状: [batch_size, num_experts, n_classes, height, width]
#         gating_outputs = self.gating_network(x)  # 形状: [batch_size, num_experts, height, width]

#         # 通过广播将gating_outputs乘以expert_outputs
#         output = torch.sum(gating_outputs.unsqueeze(2) * expert_outputs, dim=1)  # 形状: [batch_size, n_classes, height, width]
#         return torch.sigmoid(output)  # 直接返回sigmoid激活后的结果

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.unet_parts import *
# 使用 U-Net 作为门控网络
class UNetGatingNetwork(nn.Module):
    def __init__(self, input_channels, num_experts):
        super(UNetGatingNetwork, self).__init__()
        self.n_channels = input_channels
        self.n_classes = num_experts

        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, num_experts)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.softmax(logits, dim=1)
# 基于专家预测的置信度动态调整门控权重
class UncertaintyAwareFusion(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp

    def forward(self, expert_outputs, gate_weights):
        # 1. 确保 expert_outputs 的范围在 [0, 1] 内，避免溢出
        expert_outputs = torch.clamp(expert_outputs, min=1e-5, max=1 - 1e-5)  # 避免除零问题

        # 2. 计算专家预测的熵
        # 采用稳定的方式计算熵
        eps = 1e-12
        entropy = -expert_outputs * torch.log(expert_outputs + eps) - (1 - expert_outputs) * torch.log(1 - expert_outputs + eps)
        entropy = entropy.mean(dim=2)  
        #entropy = entropy.mean(dim=[3, 4])  
        # 3. 生成置信度权重
        # 使用 tanh 限制熵的范围，并调整置信度
        scaled_entropy = torch.tanh(entropy / self.temp)
        confidence = 1 - scaled_entropy

        # 4. 融合门控权重和置信度
        calibrated_weights = gate_weights * confidence

        # 5. 归一化融合后的权重
        sum_weights = torch.sum(calibrated_weights, dim=1, keepdim=True)
        calibrated_weights = calibrated_weights / (sum_weights + 1e-8)  # 避免除以接近零的值

        return calibrated_weights
    
# 在门控网络中引入域原型记忆库，通过对比学习增强跨域判别能力
# class DomainAwareGating(UNetGatingNetwork):
#     def __init__(self, input_channels, num_experts, num_domains=3):
#         super().__init__(input_channels, num_experts)
#         # 域原型记忆库 [num_domains, num_experts, feat_dim]
#         self.register_buffer('domain_prototypes', torch.randn(num_domains, num_experts, 512))
        
#         # 域分类器
#         self.domain_cls = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(512, num_domains)
#         )

#     def forward(self, x, domain_id=None):
#         # 标准门控计算
#         gate_logits = super().forward(x)
        
#         # 域感知对比学习
#         if self.training and domain_id is not None:
#             # 获取高层特征
#             feat = self.down4(self.down3(self.down2(self.down1(self.inc(x)))))
#             feat = F.adaptive_avg_pool2d(feat, (1,1)).flatten(1)
            
#             # 计算对比损失
#             prototypes = self.domain_prototypes[domain_id]  # [B,N,D]
#             logits = torch.einsum('bd,bnd->bn', feat, prototypes)
#             contrast_loss = F.cross_entropy(logits, torch.arange(len(domain_id)).to(x.device))
            
#             return gate_logits, contrast_loss
            
#         return gate_logits
    
# 在UNet门控网络输出后增加时空注意力模块，实现像素级专家权重动态校准
class SpatioTemporalRouter(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        assert num_experts > 0, "num_experts must be a positive integer"
        
        # 动态调整中间层的通道数，确保至少为1
        mid_channels = max(1, num_experts // 4)  # 如果 num_experts // 4 为0，则使用1

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 类似于 MLP 中的第一层
            nn.Conv2d(num_experts, mid_channels, 1),  # 输入通道数为 num_experts
            nn.ReLU(),
             # 类似于 MLP 中的第二层
            nn.Conv2d(mid_channels, num_experts, 1),  # 输出通道数为 num_experts
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(num_experts, 1, 3, padding=1),  # 输入通道数为 num_experts
            nn.Sigmoid()
        )

    def forward(self, gate_weights):
        # 通道注意力 [B, N, H, W] -> [B, N, 1, 1]
        ca = self.channel_att(gate_weights)
        # 空间注意力 [B, N, H, W] -> [B, 1, H, W]
        sa = self.spatial_att(gate_weights)
        # 联合校准
        calibrated = gate_weights * ca * sa
        return calibrated / calibrated.sum(dim=1, keepdim=True)

# # 修改 MixtureOfExperts，使用 UNetGatingNetwork
# class MixtureOfExperts(nn.Module):
#     def __init__(self, num_experts, input_shape, n_classes, trained_experts):
#         super(MixtureOfExperts, self).__init__()
#         self.num_experts = num_experts
#         self.experts = nn.ModuleList(trained_experts)
#         self.gating_network = UNetGatingNetwork(input_shape[0], num_experts)

#     def forward(self, x):
#         expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # 形状: [batch_size, num_experts, n_classes, height, width]
#         gating_outputs = self.gating_network(x)  # 形状: [batch_size, num_experts, height, width]

#         # 通过广播将 gating_outputs 乘以 expert_outputs
#         output = torch.sum(gating_outputs.unsqueeze(2) * expert_outputs, dim=1)  # 形状: [batch_size, n_classes, height, width]
#         return output # 直接返回 sigmoid 激活后的结果
def topk_sparse_gating(gate_weights, k=3):
    """
    gate_weights: Tensor [B, N, H, W] - soft 权重
    返回：
        sparse_weights: Tensor [B, N, H, W] - 只保留 top-k，归一化后的稀疏权重
    """
    B, N, H, W = gate_weights.shape

    # 获取 top-k 索引
    topk_vals, topk_idx = torch.topk(gate_weights, k=k, dim=1)  # [B, k, H, W]

    # 构建稀疏 one-hot mask
    mask = torch.zeros_like(gate_weights)  # [B, N, H, W]
    mask.scatter_(1, topk_idx, 1)

    # 只保留 top-k 权重
    sparse_weights = gate_weights * mask

    # 对 top-k 权重归一化（按专家维度）
    sparse_weights = sparse_weights / (sparse_weights.sum(dim=1, keepdim=True) + 1e-8)

    return sparse_weights  # [B, N, H, W]

def add_noise(x, epoch, max_epoch, noise_std=0.1):
    # 线性增大的高斯噪声扰动
    factor = (epoch + 1) / max_epoch
    noise = torch.randn_like(x) * noise_std * factor
    return x + noise

class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, input_shape, n_classes, trained_experts, num_domains=3, temp=0.5):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(trained_experts)  # 专家网络列表
        self.gating_network = UNetGatingNetwork(input_shape[0], num_experts)
        #self.gating_network = DomainAwareGating(input_shape[0], num_experts, num_domains)  # 使用 DomainAwareGating
        self.router = SpatioTemporalRouter(num_experts)  # 时空注意力模块
        self.uncertainty_fusion = UncertaintyAwareFusion(temp)  # 基于置信度的融合模块

    def forward(self, x,  epoch=None, max_epoch=None, domain_id=None):
        # 获取专家网络的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, n_classes, height, width]
        
        # # 获取门控网络的输出
        # if self.training and domain_id is not None:
        #     # 如果是训练模式且提供了 domain_id，则计算对比损失
        #     gating_outputs, contrast_loss = self.gating_network(x, domain_id)  # [batch_size, num_experts, height, width]
        # else:
        #     # 如果是推理模式或未提供 domain_id，则只获取门控输出
        #     gating_outputs = self.gating_network(x)  # [batch_size, num_experts, height, width]
        #     contrast_loss = None
        gating_outputs = self.gating_network(x)

        calibrated_weights = gating_outputs  # 初始门控权重

        # 如果在训练过程中，加入高斯噪声
        # if self.training and epoch is not None and max_epoch is not None:
        #     gating_outputs = add_noise(gating_outputs, epoch, max_epoch, noise_std=0.1)
        # 选 top-k 专家
        gating_outputs = topk_sparse_gating(gating_outputs, k=2)
        # 基于置信度动态调整门控权重
        calibrated_weights = self.uncertainty_fusion(expert_outputs, gating_outputs)  # [batch_size, num_experts, height, width]
        calibrated_weights = gating_outputs
        # 通过时空注意力模块进一步校准门控权重
        calibrated_weights = self.router(calibrated_weights)  # [batch_size, num_experts, height, width]

        # 通过广播将 calibrated_weights 乘以 expert_outputs
        output = torch.sum(calibrated_weights.unsqueeze(2) * expert_outputs, dim=1)  # [batch_size, n_classes, height, width]

        # 返回最终输出和对比损失
        return output #, contrast_loss