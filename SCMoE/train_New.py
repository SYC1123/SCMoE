import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from model.RMT_UNet import RMT_UNet
# 设置环境变量以避免某些库之间的冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch import Tensor
from tqdm import tqdm  # 导入tqdm
from model.unet_model import UNet
from dataset_New import ISBI_Loader_New
from torch import optim
import torch.nn as nn
import torch
import logging
from MoE_New import MixtureOfExperts
from torch.cuda.amp import GradScaler, autocast
# 定义Dice系数计算函数
def dice_coefficient(pred: Tensor, target: Tensor) -> float:
    smooth = 1.0  # 为了数值稳定性，避免除以零
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
def train_net_val(net, device,
                  train_data_path, val_data_path,
                  epochs=50, batch_size=2, lr=1.5e-5,
                  model_save_path=None, log_file=None,
                  model_name=None):
    import logging, os
    from tqdm import tqdm

    # —— 日志配置 ——#
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
                            logging.StreamHandler()  # 同时输出到控制台
                        ])
    logger = logging.getLogger()

    best_dice_model = os.path.join(model_save_path, f"best_dice_{model_name}.pth")
    final_model     = os.path.join(model_save_path, f"{model_name}.pth")
    # 输出训练的基本信息
    # logger.info(f"Starting training with the following parameters:")
    # logger.info(f"  Epochs: {epochs}")
    # logger.info(f"  Batch size: {batch_size}")
    # logger.info(f"  Learning rate: {lr}")
    # logger.info(f"  Train data path: {train_data_path}")
    # logger.info(f"  Validation data path: {val_data_path}")
    # logger.info(f"  Model save path: {model_save_path}")
    # logger.info(f"  Final Model save path: {final_model}")
    # logger.info(f"  Best Dice Model save path: {best_dice_model}")
    # logger.info(f"  Log file: {log_file}")
    # logger.info(f"Train MoE gating on multi-domain data")

    # —— DataLoader：return_domain=True 让 __getitem__ 返回三元组 ——#
    train_ds = ISBI_Loader_New(train_data_path, return_domain=True)
    val_ds   = ISBI_Loader_New(val_data_path,   return_domain=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    print("训练集大小：", len(train_loader.dataset))
    print("验证集大小：", len(val_loader.dataset))
    # —— 训练参数设置 ——#
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr,
                                    weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    best_dice = 0.0

    for epoch in range(1, epochs+1):
        net.train()
        running_loss = 0.0
        running_dice = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit='batch')

        for batch in pbar:
            # 根据 return_domain 自动解包
            if len(batch) == 3:
                images, labels, domain_ids = batch
                #print("domain_ids =", domain_ids)
                domain_ids = domain_ids.to(device)
            else:
                raise RuntimeError("Expected 3 items from DataLoader but got {}".format(len(batch)))

            images = images.to(device).float()
            labels = labels.to(device).float()
            # 检查标签值是否超出 [0, 1] 范围
            if not torch.all(labels >= 0) or not torch.all(labels <= 1):
                raise ValueError(f"Labels contain values outside [0, 1]: {labels.min()}, {labels.max()}")

            optimizer.zero_grad()
            # 假设 net.forward 接受 domain_id
            outputs, dom_loss = net(images, domain_id=domain_ids)
            main_loss = criterion(outputs, labels.view(-1, 1, 384, 384))  
            if dom_loss is not None and dom_loss.ndim > 0:
                dom_loss = dom_loss.mean()
            total_loss = main_loss + (0.1 * dom_loss if dom_loss is not None else 0)

            total_loss.backward()
            optimizer.step()

            dice = dice_coefficient(torch.sigmoid(outputs), labels)

            running_loss += main_loss.item()
            running_dice += dice.item()
            pbar.set_postfix(train_loss=running_loss / (pbar.n+1),
                             train_dice=running_dice / (pbar.n+1))

        avg_train_loss = running_loss / len(train_loader)
        avg_train_dice = running_dice / len(train_loader)
        logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}")

        # —— 验证 ——#
        net.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, labels, domain_ids = batch
                images = images.to(device).float()
                labels = labels.to(device).float()
                outputs, _ = net(images, domain_id=domain_ids.to(device))
                loss = criterion(outputs, labels.view(-1, 1, 384, 384))  
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        logger.info(f"[Epoch {epoch}] Val   Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")

        # 保存最优模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(net.state_dict(), best_dice_model)
            logger.info(f"Saved best dice model: {best_dice_model}")

    # 结束后保存最终模型
    torch.save(net.state_dict(), final_model)
    logger.info(f"Saved final model: {final_model}")


# 训练MoE时加载专家模型
def load_expert_model(model_path, device):
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # 如果保存时使用了 DataParallel，加载时需要移除 `module.` 前缀
    if 'module.' in list(state_dict.keys())[0]:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        return new_state_dict
    else:
        return state_dict
def train_moe_with_experts(device, data_path, expert_model_paths):
    # 创建专家模型
    expert1 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert1_state_dict = load_expert_model(expert_model_paths[0], device)
    expert1.load_state_dict(expert1_state_dict)

    expert2 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert2_state_dict = load_expert_model(expert_model_paths[1], device)
    expert2.load_state_dict(expert2_state_dict)

    expert3 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert3_state_dict = load_expert_model(expert_model_paths[2], device)
    expert3.load_state_dict(expert3_state_dict)

    expert4 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert4_state_dict = load_expert_model(expert_model_paths[3], device)
    expert4.load_state_dict(expert4_state_dict)

    expert5 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert5_state_dict = load_expert_model(expert_model_paths[4], device)
    expert5.load_state_dict(expert5_state_dict)

    # for param in expert1.parameters():
    #     param.requires_grad = False
    # for param in expert2.parameters():
    #     param.requires_grad = False
    # for param in expert3.parameters():
    #     param.requires_grad = False
    # for param in expert4.parameters():
    #     param.requires_grad = False
    # for param in expert5.parameters():
    #     param.requires_grad = False  
    # experts = [expert1, expert2, expert3, expert4, expert5]
    # for expert in experts:
    #     expert.to(device)  #
    # 将专家模型传递给MoE
    net = MixtureOfExperts(num_experts, input_shape,  [expert1, expert2, expert3, expert4, expert5])
    # 使用 DataParallel 分发到多 GPU（如果你使用多个 GPU）
    net = nn.DataParallel(net)
    # 移动模型到指定设备
    net = net.to(device=device)
    #print(next(net.parameters()).device)  # 查看模型权重所在的设备



    # 训练MoE网络
    train_net_val()

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(device)
    # # 加载网络，图片单通道1，分类为1。
    # net = UNet(n_channels=1, n_classes=1)
    num_experts = 5  # 可以根据需要调整专家的数量
    #input_shape = (1, 240, 240)  # 根据你的数据集调整输入形状
    input_shape = (1,384,384)
    train_moe_with_experts()