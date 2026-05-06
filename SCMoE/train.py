import os

from model.RMT_UNet import RMT_UNet

# 设置环境变量以避免某些库之间的冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm  # 导入tqdm
from model.unet_model import UNet
from dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
import logging
from MoE import MixtureOfExperts
from torch.cuda.amp import GradScaler, autocast
# Dice Loss（Soft Dice，适用于训练）
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# 定义Dice系数计算函数
def dice_coefficient(pred: Tensor, target: Tensor) -> float:
    smooth = 1.0  # 为了数值稳定性，避免除以零
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 辅助函数：计算边界距离图
def compute_distance_map(mask):
    mask = mask.cpu().numpy()
    dist_map = np.zeros_like(mask, dtype=np.float32)
    for i in range(mask.shape[0]):
        posmask = mask[i].astype(np.bool_)
        negmask = ~posmask
        dist_out = ndi.distance_transform_edt(negmask)
        dist_in = ndi.distance_transform_edt(posmask)
        dist_map[i] = dist_out + dist_in
    return torch.from_numpy(dist_map).to(mask.device)

# Boundary Loss
def boundary_loss(pred, target):
    pred = pred.float()
    target = target.float()
    distance_map = compute_distance_map(target.squeeze(1)).unsqueeze(1)  # [B, 1, H, W]
    multipled = pred * distance_map
    loss = multipled.mean()
    return loss

def train_net(net, device, data_path, epochs=50, batch_size=16, lr=0.000015, model_save_path=None, log_file=None, model_name=None):
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
                            logging.StreamHandler()  # 同时输出到控制台
                        ])
    logger = logging.getLogger()
    best_loss_model_save_path = os.path.join(model_save_path,  f"best_loss_model_{model_name}.pth")
    model_save_path = os.path.join(model_save_path, f"{model_name}.pth")
    # os.makedirs(model_save_path, exist_ok=True)
    # os.makedirs(best_loss_model_save_path, exist_ok=True)
    # logger.info(f"Starting training with the following parameters:")
    # logger.info(f"  Epochs: {epochs}")
    # logger.info(f"  Batch size: {batch_size}")
    # logger.info(f"  Learning rate: {lr}")
    # logger.info(f"  Data path: {data_path}")
    # logger.info(f"  Model save path: {model_save_path}")
    # logger.info(f"  Best Loss Model save path: {best_loss_model_save_path}")
    # logger.info(f"  Log file: {log_file}")


    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    best_dice = 0.0
    total_loss = 0
    total_dice = 0
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 使用tqdm创建进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        # 按照batch_size开始训练
        for image, label in progress_bar:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = torch.sigmoid(net(image))  # 确保pred是二进制的
            # pred形状 [batch_size,n_classes,width,height]
            # 计算loss
            loss = criterion(pred, label)
            total_loss += loss.item()
            # 计算Dice系数
            dice = dice_coefficient(pred, label)
            total_dice += dice.item()
            progress_bar.set_postfix(loss=loss.item(), dice=dice)  # 更新进度条的后缀
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(),best_loss_model_save_path)
            # 更新参数
            loss.backward()
            optimizer.step()
        # 每个epoch结束后输出平均loss和Dice系数
        average_loss = total_loss / len(train_loader)
        average_dice = total_dice / len(train_loader)
        logger.info(f'Epoch {epoch+1}, Average Loss: {average_loss:.4f}, Average Dice: {average_dice:.4f}')
        total_loss = 0
        total_dice = 0
    # 在训练完专家模型后保存模型参数
    if model_save_path:
        torch.save(net.state_dict(), model_save_path)
        logger.info(f"Expert model saved to {model_save_path}")
# def train_net_val(net, device, train_data_path, val_data_path, epochs=50, batch_size=2, lr=0.000015, model_save_path=None, log_file=None, model_name=None):
#     # 配置日志
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s - %(levelname)s - %(message)s',
#                         handlers=[
#                             logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
#                             logging.StreamHandler()  # 同时输出到控制台
#                         ])
#     logger = logging.getLogger()
#     best_loss_model_save_path = os.path.join(model_save_path, f"best_loss_model_{model_name}.pth")
#     best_dice_model_save_path = os.path.join(model_save_path, f"best_dice_model_{model_name}.pth")
#     model_save_path = os.path.join(model_save_path, f"{model_name}.pth")
#     # 输出训练的基本信息
#     logger.info(f"Starting training with the following parameters:")
#     logger.info(f"  Epochs: {epochs}")
#     logger.info(f"  Batch size: {batch_size}")
#     logger.info(f"  Learning rate: {lr}")
#     logger.info(f"  Train data path: {train_data_path}")
#     logger.info(f"  Validation data path: {val_data_path}")
#     logger.info(f"  Model save path: {model_save_path}")
#     logger.info(f"  Best Loss Model save path: {best_loss_model_save_path}")
#     logger.info(f"  Best Dice Model save path: {best_dice_model_save_path}")
#     logger.info(f"  Log file: {log_file}")

#     # 加载训练集
#     train_dataset = ISBI_Loader(train_data_path)
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True)

#     # 加载验证集
#     val_dataset = ISBI_Loader(val_data_path)
#     val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False)

#     # 定义RMSprop算法
#     optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#     # 定义Loss算法
#     criterion = nn.BCEWithLogitsLoss()
#     # best_loss统计，初始化为正无穷
#     best_loss = float('inf')
#     best_dice = 0.0
#     total_loss = 0
#     total_dice = 0
#     # 训练epochs次
#     for epoch in range(epochs):
#         # 训练模式
#         net.train()
#         # 使用tqdm创建进度条
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
#         # 按照batch_size开始训练
#         for image, label in progress_bar:
#             optimizer.zero_grad()
#             # 将数据拷贝到device中
#             image = image.to(device=device, dtype=torch.float32)
#             label = label.to(device=device, dtype=torch.float32)
#             # 使用网络参数，输出预测结果
#             pred = torch.sigmoid(net(image, epoch=epoch, max_epoch=epochs))   # 确保pred是二进制的
#             # pred形状 [batch_size,n_classes,width,height]
#             # 计算loss
#             loss = criterion(pred, label)
#             total_loss += loss.item()
#             # 计算Dice系数
#             dice = dice_coefficient(pred, label)
#             total_dice += dice.item()
#             progress_bar.set_postfix(loss=loss.item(), dice=dice)  # 更新进度条的后缀
#             # # 保存loss值最小的网络参数
#             # if loss < best_loss:
#             #     best_loss = loss
#             #     torch.save(net.state_dict(), best_loss_model_save_path)
#             # 更新参数
#             loss.backward()
#             optimizer.step()
#         # 每个epoch结束后输出平均loss和Dice系数
#         average_loss = total_loss / len(train_loader)
#         average_dice = total_dice / len(train_loader)
#         logger.info(f'Epoch {epoch+1}, Average Loss: {average_loss:.4f}, Average Dice: {average_dice:.4f}')
#         total_loss = 0
#         total_dice = 0

#         # 验证模式
#         net.eval()
#         val_loss = 0
#         val_dice = 0
#         with torch.no_grad():
#             for image, label in val_loader:
#                 image = image.to(device=device, dtype=torch.float32)
#                 label = label.to(device=device, dtype=torch.float32)
#                 pred = torch.sigmoid(net(image))
#                 loss = criterion(pred, label)
#                 val_loss += loss.item()
#                 val_dice += dice_coefficient(pred, label).item()
#         average_val_loss = val_loss / len(val_loader)
#         average_val_dice = val_dice / len(val_loader)
#         logger.info(f'Epoch {epoch+1}, Validation Loss: {average_val_loss:.4f}, Validation Dice: {average_val_dice:.4f}')

#         # 保存最佳验证Dice系数的模型
#         if average_val_dice > best_dice:
#             best_dice = average_val_dice
#             torch.save(net.state_dict(), best_dice_model_save_path)

#     # 在训练完专家模型后保存模型参数
#     if model_save_path:
#         torch.save(net.state_dict(), model_save_path)
#         logger.info(f"Expert model saved to {model_save_path}")

def train_net_val(net, device, train_data_path, val_data_path, epochs=50, batch_size=2, lr=0.000015,
                  model_save_path=None, log_file=None, model_name=None, dice_weight=1.0):

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()

    best_loss_model_save_path = os.path.join(model_save_path, f"best_loss_model_{model_name}.pth")
    best_dice_model_save_path = os.path.join(model_save_path, f"best_dice_model_{model_name}.pth")
    final_model_save_path = os.path.join(model_save_path, f"{model_name}.pth")

    logger.info(f"Starting training with the following parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Train data path: {train_data_path}")
    logger.info(f"  Validation data path: {val_data_path}")
    logger.info(f"  Model save path: {model_save_path}")
    logger.info(f"  Best Loss Model save path: {best_loss_model_save_path}")
    logger.info(f"  Best Dice Model save path: {best_dice_model_save_path}")
    logger.info(f"  Log file: {log_file}")

    # 加载数据
    train_dataset = ISBI_Loader(train_data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ISBI_Loader(val_data_path, apply_transform=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    best_dice = 0.0

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        total_dice = 0
        progress_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}", unit='batch')

        for image, label in progress_bar:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output_logits = net(image, epoch, epochs)
            pred = torch.sigmoid(output_logits)

            boundary = boundary_loss(pred, label)
            dice = dice_loss(pred, label)
            loss = boundary + dice_weight * dice

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(pred, label).item()

            progress_bar.set_postfix(loss=loss.item(), dice=(1 - dice.item()))

        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        logger.info(f"[Train] Epoch {epoch+1}: Average Loss = {avg_loss:.4f}, Average Dice = {avg_dice:.4f}")

        # 验证
        net.eval()
        val_loss = 0
        val_dice = 0
        val_bar = tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{epochs}", unit='batch')

        with torch.no_grad():
            for image, label in val_bar:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                output_logits = net(image)
                pred = torch.sigmoid(output_logits)

                boundary = boundary_loss(pred, label)
                dice = dice_loss(pred, label)
                loss = boundary + dice_weight * dice

                val_loss += loss.item()
                val_dice += dice_coefficient(pred, label).item()

                val_bar.set_postfix(loss=loss.item(), dice=(1 - dice.item()))

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        logger.info(f"[Val] Epoch {epoch+1}: Average Loss = {avg_val_loss:.4f}, Average Dice = {avg_val_dice:.4f}")

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(net.state_dict(), best_dice_model_save_path)
            logger.info(f"Best Dice model saved at epoch {epoch+1}")

    torch.save(net.state_dict(), final_model_save_path)
    logger.info(f"Final model saved to {final_model_save_path}")
def train_net_val_BCE(net, device, train_data_path, val_data_path, epochs=50, batch_size=2, lr=0.000015,
                  model_save_path=None, log_file=None, model_name=None, dice_weight=1.0):
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()

    best_loss_model_save_path = os.path.join(model_save_path, f"best_loss_model_{model_name}.pth")
    best_dice_model_save_path = os.path.join(model_save_path, f"best_dice_model_{model_name}.pth")
    final_model_save_path = os.path.join(model_save_path, f"{model_name}.pth")

    logger.info(f"Starting training with the following parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Train data path: {train_data_path}")
    logger.info(f"  Validation data path: {val_data_path}")
    logger.info(f"  Model save path: {model_save_path}")
    logger.info(f"  Best Loss Model save path: {best_loss_model_save_path}")
    logger.info(f"  Best Dice Model save path: {best_dice_model_save_path}")
    logger.info(f"  Log file: {log_file}")

    # 加载数据
    train_dataset = ISBI_Loader(train_data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ISBI_Loader(val_data_path, apply_transform=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    bce_criterion = nn.BCEWithLogitsLoss()

    best_dice = 0.0

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        total_dice = 0
        progress_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}", unit='batch')

        for image, label in progress_bar:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output_logits = net(image, epoch, epochs)
            pred = torch.sigmoid(output_logits)

            bce = bce_criterion(pred, label)
            dice = dice_loss(pred, label)
            loss = bce + dice_weight * dice

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(pred, label).item()

            progress_bar.set_postfix(loss=loss.item(), dice=(1 - dice.item()))

        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        logger.info(f"[Train] Epoch {epoch+1}: Average Loss = {avg_loss:.4f}, Average Dice = {avg_dice:.4f}")

        # 验证
        net.eval()
        val_loss = 0
        val_dice = 0
        val_bar = tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{epochs}", unit='batch')

        with torch.no_grad():
            for image, label in val_bar:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                output_logits = net(image)
                pred = torch.sigmoid(output_logits)

                bce = bce_criterion(pred, label)
                dice = dice_loss(pred, label)
                loss = bce + dice_weight * dice

                val_loss += loss.item()
                val_dice += dice_coefficient(pred, label).item()

                val_bar.set_postfix(loss=loss.item(), dice=(1 - dice.item()))

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        logger.info(f"[Val] Epoch {epoch+1}: Average Loss = {avg_val_loss:.4f}, Average Dice = {avg_val_dice:.4f}")

        # 保存最佳模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(net.state_dict(), best_dice_model_save_path)
            logger.info(f"Best Dice model saved at epoch {epoch+1}")

    # 保存最终模型
    torch.save(net.state_dict(), final_model_save_path)
    logger.info(f"Final model saved to {final_model_save_path}")

def train_net_val_BCE_expert(net, device, train_data_path, val_data_path, epochs=50, batch_size=2, lr=0.000015,
                  model_save_path=None, log_file=None, model_name=None, dice_weight=1.0):
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()

    best_loss_model_save_path = os.path.join(model_save_path, f"best_loss_model_{model_name}.pth")
    best_dice_model_save_path = os.path.join(model_save_path, f"best_dice_model_{model_name}.pth")
    final_model_save_path = os.path.join(model_save_path, f"{model_name}.pth")

    logger.info(f"Starting training with the following parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Train data path: {train_data_path}")
    logger.info(f"  Validation data path: {val_data_path}")
    logger.info(f"  Model save path: {model_save_path}")
    logger.info(f"  Best Loss Model save path: {best_loss_model_save_path}")
    logger.info(f"  Best Dice Model save path: {best_dice_model_save_path}")
    logger.info(f"  Log file: {log_file}")

    # 加载数据
    train_dataset = ISBI_Loader(train_data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ISBI_Loader(val_data_path, apply_transform=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    bce_criterion = nn.BCEWithLogitsLoss()

    best_dice = 0.0

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        total_dice = 0
        progress_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{epochs}", unit='batch')

        for image, label in progress_bar:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            output_logits = net(image)
            pred = torch.sigmoid(output_logits)
            print(pred.min(), pred.max(), pred.mean())
            bce = bce_criterion(pred, label)
            dice = dice_loss(pred, label)
            loss = bce + dice_weight * dice

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(pred, label).item()

            progress_bar.set_postfix(loss=loss.item(), dice=(1 - dice.item()))

        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        logger.info(f"[Train] Epoch {epoch+1}: Average Loss = {avg_loss:.4f}, Average Dice = {avg_dice:.4f}")

        # 验证
        net.eval()
        val_loss = 0
        val_dice = 0
        val_bar = tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{epochs}", unit='batch')

        with torch.no_grad():
            for image, label in val_bar:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                output_logits = net(image)
                pred = torch.sigmoid(output_logits)
                print(pred.min(), pred.max(), pred.mean())
                bce = bce_criterion(pred, label)
                dice = dice_loss(pred, label)
                loss = bce + dice_weight * dice

                val_loss += loss.item()
                val_dice += dice_coefficient(pred, label).item()

                val_bar.set_postfix(loss=loss.item(), dice=(1 - dice.item()))

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        logger.info(f"[Val] Epoch {epoch+1}: Average Loss = {avg_val_loss:.4f}, Average Dice = {avg_val_dice:.4f}")

        # 保存最佳模型
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(net.state_dict(), best_dice_model_save_path)
            logger.info(f"Best Dice model saved at epoch {epoch+1}")

    # 保存最终模型
    torch.save(net.state_dict(), final_model_save_path)
    logger.info(f"Final model saved to {final_model_save_path}")

def train_net_val_expert(net, device, data_path, epochs=150, batch_size=32, lr=0.000015
                         , model_save_path=None, log_file=None, model_name=None, val_split=0.2, dice_weight=0.2):
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    
    best_loss_model_save_path = os.path.join(model_save_path, f"best_loss_model_{model_name}.pth")
    best_dice_model_save_path = os.path.join(model_save_path, f"best_dice_model_{model_name}.pth")
    model_save_path = os.path.join(model_save_path, f"{model_name}.pth")
    
    # 输出训练的基本信息
    logger.info(f"Starting training with the following parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Validation split: {val_split}")
    logger.info(f"  Model save path: {model_save_path}")
    logger.info(f"  Best Loss Model save path: {best_loss_model_save_path}")
    logger.info(f"  Best Dice Model save path: {best_dice_model_save_path}")
    logger.info(f"  Log file: {log_file}")

    # 加载数据集
    # dataset1 = ISBI_Loader(data_path, apply_transform=True)
    # dataset2 = ISBI_Loader(data_path, apply_transform=True)
    # #dataset3 = ISBI_Loader(data_path, apply_transform=True)
    # print("数据集1大小：", len(dataset1))
    # dataset_origin = ISBI_Loader(data_path, apply_transform=False)
    # dataset = ConcatDataset([dataset1, dataset2, dataset_origin])
    # print("合并后数据集大小：", len(dataset))
    dataset = ISBI_Loader(data_path, apply_transform= False)
    # 划分训练集和验证集
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器和损失函数
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    best_dice = 0.0

    for epoch in range(epochs):
        net.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        total_loss = 0
        total_dice = 0

        for image, label in progress_bar:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = torch.sigmoid(net(image))
            bce_loss = criterion(pred, label)
            dice = dice_loss(pred, label)
            loss = bce_loss + dice_weight * dice
            total_loss += loss.item()
            dice = dice_coefficient(pred, label)
            total_dice += dice.item()
            progress_bar.set_postfix(loss=loss.item(), dice=dice.item())
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_loader)
        average_dice = total_dice / len(train_loader)
        logger.info(f'Epoch {epoch+1}, Average Loss: {average_loss:.4f}, Average Dice: {average_dice:.4f}')

        net.eval()
        val_loss = 0
        val_dice = 0
        val_bar = tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/{epochs}", unit='batch')
        with torch.no_grad():
            for image, label in val_bar:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = torch.sigmoid(net(image))
                bce_loss = criterion(pred, label)
                dice = dice_loss(pred, label)
                loss = bce_loss + dice_weight * dice
                val_loss += loss.item()
                val_dice += dice_coefficient(pred, label).item()
        average_val_loss = val_loss / len(val_loader)
        average_val_dice = val_dice / len(val_loader)
        logger.info(f'Epoch {epoch+1}, Validation Loss: {average_val_loss:.4f}, Validation Dice: {average_val_dice:.4f}')

        if average_val_dice > best_dice:
            best_dice = average_val_dice
            torch.save(net.state_dict(), best_dice_model_save_path)

    if model_save_path:
        torch.save(net.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")
def validate(net, device, val_loader, criterion, use_amp=False):
    net.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = net(image)
                loss = criterion(pred, label)
            
            pred = torch.sigmoid(pred)
            dice = dice_coefficient(pred, label)
            
            total_loss += loss.item()
            total_dice += dice.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    return avg_loss, avg_dice

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
def train_and_save_experts(device, data_path):
    expert1 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert1 = nn.DataParallel(expert1,device_ids=[0, 1])
    expert1.to(device=device)
    train_net(expert1, device, data_path[0], model_save_path="expert1.pth",log_file="train_expert1_log.txt")

    expert2 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert2 = nn.DataParallel(expert2,device_ids=[0, 1])
    expert2.to(device=device)
    train_net(expert2, device, data_path[1], model_save_path="expert2.pth",log_file="train_expert2_log.txt")

    expert3 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert3 = nn.DataParallel(expert3,device_ids=[0, 1])
    expert3.to(device=device)
    train_net(expert3, device, data_path[2], model_save_path="expert3.pth",log_file="train_expert3_log.txt")

    return "expert1.pth", "expert2.pth", "expert3.pth"
def train_and_save_single_expert(device, data_path, model_name = None, model_save_path = None, log_file = None, val_data_path = None):
    expert1 = RMT_UNet(n_channels=1, n_classes=n_classes)
    expert1 = nn.DataParallel(expert1)
    expert1.to(device=device)

    train_net_val_BCE_expert(expert1, device, data_path,val_data_path=val_data_path,epochs=15, batch_size=12, lr=0.000125, model_name=model_name,model_save_path=model_save_path,log_file=log_file)

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

    for param in expert1.parameters():
        param.requires_grad = False
    for param in expert2.parameters():
        param.requires_grad = False
    for param in expert3.parameters():
        param.requires_grad = False
    for param in expert4.parameters():
        param.requires_grad = False
    for param in expert5.parameters():
        param.requires_grad = False    
    # 将专家模型传递给MoE
    net = MixtureOfExperts(num_experts, input_shape, n_classes, [expert1, expert2, expert3, expert4, expert5])
    
    # 移动模型到指定设备
    net = net.to(device=device)
    print(next(net.parameters()).device)  # 查看模型权重所在的设备

    # 使用 DataParallel 分发到多 GPU（如果你使用多个 GPU）
    net = nn.DataParallel(net)

    # 训练MoE网络
    train_net_val_BCE(net, device, data_path, epochs=120, batch_size=12, lr=0.000125, model_name="MoE", log_file="train_MoE_log.txt")

def train_moe_with_experts_Domain(device, data_path, expert_model_paths, val_data_path = None, model_name = None, log_file = None):
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


    for param in expert1.parameters():
        param.requires_grad = False
    for param in expert2.parameters():
        param.requires_grad = False
    for param in expert3.parameters():
        param.requires_grad = False 
    # 将专家模型传递给MoE
    net = MixtureOfExperts(3, input_shape, n_classes, [expert1, expert2, expert3])
    
    # 移动模型到指定设备
    net = net.to(device=device)
    print(next(net.parameters()).device)  # 查看模型权重所在的设备

    # 使用 DataParallel 分发到多 GPU（如果你使用多个 GPU）
    net = nn.DataParallel(net)

    # 训练MoE网络
    train_net_val_BCE(net, device, data_path, val_data_path = val_data_path ,epochs=30, batch_size=12, lr=0.000125, model_name=model_name,model_save_path="/",log_file="train_MoE_new_log.txt")

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(device)
    # # 加载网络，图片单通道1，分类为1。
    # net = UNet(n_channels=1, n_classes=1)
    num_experts = 5  # 可以根据需要调整专家的数量
    #input_shape = (1, 240, 240)  # 根据你的数据集调整输入形状
    input_shape = (1,384,384)
    n_classes = 2

    # 下面是训练MoE的代码

    train_moe_with_experts(device, data_path="MoE-test", model_name="MoE-test")

















    # # 训练模型调用下面的代码：
    # # 训练专家模型：
    # expert1 = RMT_UNet(n_channels=1, n_classes=n_classes)
    # expert1 = nn.DataParallel(expert1)
    # train_net(expert1, device, data_path1)
    # expert2 = RMT_UNet(n_channels=1, n_classes=n_classes)
    # expert2 = nn.DataParallel(expert2)
    # rain_net(expert2, device, data_path2)
    # expert3 = RMT_UNet(n_channels=1, n_classes=n_classes)
    # expert3 = nn.DataParallel(expert3)
    # train_net(expert3, device, data_path3)
    # # 训练门控网络：
    # net = MixtureOfExperts(num_experts, input_shape, n_classes, [expert1, expert2, expert3])
    # # 2. 移动模型到指定设备上
    # net = net.to(device=device)
    # print(next(net.parameters()).device)  # 查看模型权重所在的设备

    # # 3. 使用 DataParallel 分发到多 GPU（如果你使用多个 GPU）
    # net = nn.DataParallel(net)

    # train_net(net, device, data_path)
    # expert1 = RMT_UNet(n_channels=1, n_classes=n_classes)
    # expert1.to(device=device)
    # train_net_new(expert1, device, data_path)
    # expert2 = UNet(n_channels=1, n_classes=n_classes)
    # expert2.to(device=device)
    # train_net(expert2, device, data_path)
    # expert3 = UNet(n_channels=1, n_classes=n_classes)
    # expert3.to(device=device)
    # train_net(expert3, device, data_path)
    # net = MixtureOfExperts(num_experts, input_shape, n_classes, [expert1, expert2, expert3])
    # # 将网络拷贝到deivce中
    # net.to(device=device)
    # # 指定训练集地址，开始训练
    #
    # train_net(net, device, data_path)


