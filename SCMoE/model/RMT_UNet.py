import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# DFU 特征扰动模块
# class DFUModule(nn.Module):
#     def __init__(self, num_channels, momentum=0.1, std_scale=0.1):
#         super().__init__()
#         self.register_buffer('ema_mu', torch.zeros(1, num_channels, 1, 1))
#         self.register_buffer('ema_std', torch.ones(1, num_channels, 1, 1))
#         self.momentum = momentum
#         self.std_scale = std_scale

#     def forward(self, x):
#         # 计算当前 batch 的统计
#         mu = x.mean(dim=[0, 2, 3], keepdim=True)
#         std = x.std(dim=[0, 2, 3], keepdim=True)

#         if self.training:
#             # 更新 EMA
#             self.ema_mu = self.momentum * mu + (1 - self.momentum) * self.ema_mu
#             self.ema_std = self.momentum * std + (1 - self.momentum) * self.ema_std

#         # 从历史均值/方差构造扰动风格
#         noise_mu = torch.randn_like(self.ema_mu) * self.std_scale * self.ema_std
#         noise_std = torch.randn_like(self.ema_std) * self.std_scale * self.ema_std

#         mu_new = self.ema_mu + noise_mu
#         std_new = self.ema_std + noise_std
#         std_new = torch.clamp(std_new, min=1e-6)

#         # 应用扰动
#         x_norm = (x - mu) / (std + 1e-6)
#         x_new = x_norm * std_new + mu_new
#         return x_new

# class UTTA_Module(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.num_classes = num_classes

#     def forward(self, output, feat):
#         # 1. evidential uncertainty
#         evidence = torch.exp(output * self.num_classes)
#         alpha = evidence + 1
#         S = torch.sum(alpha, dim=1, keepdim=True)
#         uncertainty = self.num_classes / S  # Bx1xHxW

#         # 2. get hard prediction
#         pred_class = torch.argmax(output, dim=1, keepdim=True)  # Bx1xHxW

#         # 3. build weight volume from confident features
#         feat = F.normalize(feat, dim=1)
#         weight_volume = []
#         for c in range(self.num_classes):
#             mask = (pred_class == c) * (1 - uncertainty)
#             mask = F.interpolate(mask.float(), size=feat.shape[2:], mode="bilinear", align_corners=True)
#             weight = (feat * mask).mean(dim=(0, 2, 3), keepdim=True)
#             weight = F.normalize(weight, dim=1)
#             weight_volume.append(weight)
#         weight_volume = torch.cat(weight_volume, dim=0)  # CxF×1×1

#         # 4. apply dynamic classifier
#         out2 = F.conv2d(feat, weight_volume, bias=None)
#         out2 = F.interpolate(out2, size=output.shape[2:], mode='bilinear', align_corners=True)

#         # 5. sigmoid + fusion
#         output = torch.sigmoid(output)
#         out2 = torch.sigmoid(out2)
#         fused_output = 0.8 * output + 0.2 * out2
#         return fused_output


# RMT 相关模块
# 将输入图像打块并映射到低维嵌入空间，用于后续注意力计算
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        # 通过两个 卷积层和激活函数（GELU）以及归一化层（BatchNorm2d）来提取图像特征
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  
        x = self.norm(x)
        return x


class RetNetRelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads    
        decay_init = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads
        ))
        self.decay = nn.Parameter(decay_init)        
        self.register_buffer('angle', angle)

    def generate_2d_decay(self, H: int, W: int):
        index_h = torch.arange(H).to(self.decay.device) 
        index_w = torch.arange(W).to(self.decay.device)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  
        mask = grid[:, None, :] - grid[None, :, :] 
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None] 
        return mask

    def forward(self, slen: Tuple[int]):

        with torch.no_grad():
            mask = self.generate_2d_decay(slen[0], slen[1])
        return mask

# 实现核心的自注意力机制
class VisionRetentionAll(nn.Module):
    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        # 初始化嵌入维度、注意力头数、值因子等参数
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        # 调制项
        # self.modulate_fc = nn.ModuleList([
        # nn.Linear(embed_dim, 1) for _ in range(num_heads)])
        # 查询、键、值的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos):

        bsz, h, w, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        k *= self.scaling
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  #
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  
        k = k.flatten(2, 3) 
        v = v.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)  
        # mod_factors = []
        # for h in range(self.num_heads):
        #     gate = torch.sigmoid(self.modulate_fc[h](x))  
        #     mod_factors.append(gate)
        # mod_factors = torch.stack(mod_factors, dim=1)  
        # mod_factors = mod_factors.flatten(2, 3)  
        
        # 计算查询和键的点积，得到注意力分数
        qk_mat = q @ k.transpose(-1, -2)  
        # qk_mat = qk_mat * mod_factors  
        qk_mat = qk_mat + rel_pos  
        qk_mat = torch.softmax(qk_mat, -1)  
        output = torch.matmul(qk_mat, v)  
        output = output.view(bsz, h, w, -1)  
        output = self.out_proj(output)
        return output

# 构建模型的一个基本层，包含空间衰减矩阵生成和多个自注意力块
class BasicLayer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, init_value, heads_range):
        super().__init__()
        self.Relpos = RetNetRelPos2d(embed_dim, num_heads, init_value, heads_range)
        self.blocks = nn.ModuleList([
            VisionRetentionAll(embed_dim, num_heads) for _ in range(depth)
        ])

    def forward(self, x):
        b, h, w, _ = x.size()
        rel_pos = self.Relpos((h, w))
        # 对每个自注意力块进行前向传播，输入包括特征图和空间衰减矩阵
        for blk in self.blocks:
            x = blk(x, rel_pos)
        return x


# U-Net 相关模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #print(x.shape)
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        #print(x1.shape)
        #print(x2.shape)
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        #print(x.shape)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class ManhattanAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=embed_dim, embed_dim=embed_dim)
        self.rmt_layer = BasicLayer(embed_dim=embed_dim, depth=1, num_heads=num_heads, init_value=initial_value, heads_range=heads_range)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.rmt_layer(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class RMT_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(RMT_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.dfu = DFUModule(num_channels=512)  # 对应 x5 的通道数
        # self.utta = UTTA_Module(num_classes=n_classes)


        # U-Net Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # RMT Encoder
        self.patch_embed = PatchEmbed(in_chans=512, embed_dim=32)
        self.rmt_encoder = nn.Sequential(
            BasicLayer(embed_dim=32, depth=1, num_heads=2, init_value=2, heads_range=3),
            BasicLayer(embed_dim=32, depth=1, num_heads=2, init_value=2, heads_range=3),
            BasicLayer(embed_dim=32, depth=1, num_heads=4, init_value=2, heads_range=3),
            BasicLayer(embed_dim=32, depth=1, num_heads=8, init_value=2, heads_range=3)
        )

        # Attention Gates for Skip Connections
        self.attention_gate1 = ManhattanAttention(embed_dim=64, num_heads=2, initial_value=2, heads_range=3)
        self.attention_gate2 = ManhattanAttention(embed_dim=128, num_heads=2, initial_value=2, heads_range=3)
        self.attention_gate3 = ManhattanAttention(embed_dim=256, num_heads=4, initial_value=2, heads_range=3)
        self.attention_gate4 = ManhattanAttention(embed_dim=512, num_heads=8, initial_value=2, heads_range=3)
        
        # U-Net Decoder
        self.up1 = Up(512 + 544, 256, bilinear)  # 输入维度为 RMT 输出 + U-Net 编码器的特征
        self.up2 = Up(256 + 256, 128, bilinear)
        self.up3 = Up(128 + 128, 64, bilinear)
        self.up4 = Up(64 + 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # U-Net Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # if self.training:  # 只在训练阶段扰动
        #     x5 = self.dfu(x5)

        # print(f"x1 shape: {x1.shape}")  # 检查x1尺寸
        # print(f"x2 shape: {x2.shape}")  # 检查x2尺寸
        # print(f"x3 shape: {x3.shape}")  # 检查x3尺寸
        # print(f"x4 shape: {x4.shape}")  # 检查x4尺寸
        # Apply Attention Gates to Encoder Features
        x1_attn = self.attention_gate1(x1)
        x1_attn = F.interpolate(x1_attn, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x2_attn = self.attention_gate2(x2)
        x2_attn = F.interpolate(x2_attn, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x3_attn = self.attention_gate3(x3)
        x3_attn = F.interpolate(x3_attn, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x4_attn = self.attention_gate4(x4)
        x4_attn = F.interpolate(x4_attn, size=x4.shape[2:], mode='bilinear', align_corners=True)
        # x1_attn = self.attention_gate1(x1)
        # print(f"x1_attn shape: {x1_attn.shape}")  # 检查x1_attn尺寸
        # x2_attn = self.attention_gate2(x2)
        # print(f"x2_attn shape: {x2_attn.shape}")  # 检查x2_attn尺寸
        # x3_attn = self.attention_gate3(x3)
        # print(f"x3_attn shape: {x3_attn.shape}")  # 检查x3_attn尺寸
        # x4_attn = self.attention_gate4(x4)
        # print(f"x4_attn shape: {x4_attn.shape}")  # 检查x4_attn尺寸
        # RMT Encoder
        x_rmt = self.patch_embed(x5)  # 输入 U-Net 编码器的输出
        x_rmt = self.rmt_encoder(x_rmt)
        x_rmt = x_rmt.permute(0, 3, 1, 2)  # 转换为 (B, C, H, W) 格式

        # 将 RMT 的输出与 U-Net 编码器的特征拼接
        x_rmt = F.interpolate(x_rmt, size=x5.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x_rmt, x5], dim=1)  # 拼接 RMT 输出和 U-Net 编码器的特征
        #print(x.shape)
        # U-Net Decoder with Attention-Gated Skip Connections
        x = self.up1(x, x4_attn)
        x = self.up2(x, x3_attn)
        x = self.up3(x, x2_attn)
        x = self.up4(x, x1_attn)
        logits = self.outc(x)

        # # 推理阶段启用 UTTA
        # if not self.training:
        #     logits = self.utta(logits, x_rmt)  # x_rmt 是 RMT 输出特征，作为动态分类特征

        return logits


# Example usage
if __name__ == "__main__":
    model = RMT_UNet(n_channels=3, n_classes=2, bilinear=True)
    input_tensor = torch.randn(1, 3, 384, 384)
    output = model(input_tensor)
    print(output.shape)  # Expected output shape: (1, n_classes, H, W)