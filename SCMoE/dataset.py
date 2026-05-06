import os

import numpy as np
import pandas as pd
from bezier_curve_new import apply_bezier_gray_transform

# 设置环境变量以避免某些库之间的冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import cv2
import glob
from torch.utils.data import Dataset
import random

def create_binary_mask(label_path, threshold=64):
    # 读取标签图像
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
    # 二值化处理，只保留黑色区域
    _, binary_mask = cv2.threshold(label, threshold, 1, cv2.THRESH_BINARY)
    
    return binary_mask

class ISBI_Loader(Dataset):
    def __init__(self, data_path, apply_transform=False):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # 使用glob递归搜索所有子文件夹中的.png文件
        self.imgs_path = glob.glob(os.path.join(data_path, '**/image/**/*.png'), recursive=True)
        #self.imgs_path = glob.glob(os.path.join(data_path, 'image', '*.png'))
        self.apply_transform = apply_transform
        
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # 创建二值掩码
        #binary_mask = create_binary_mask(label_path)

        # 统一图像尺寸
        target_size = (384, 384)
        image = cv2.resize(image, target_size)
        label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)
        image = image/255.0  # 归一化到0和1
        label = label / 255
        # ------------------- 下面注释的内容是使用论文中的贝塞尔曲线进行深度变换 --------------------------
        # 应用非线性变换
        # 创建一个与图像同样大小的输出图像
        # transformed_image = np.zeros_like(image, dtype=np.float32)  # 使用浮点数以防止溢出
        # # 创建掩码，只保留非黑色像素
        # mask = image > 0
        # non_black_pixels = image[mask]  # 获取非黑色像素值
        # # 应用非线性变换
        # transformed_pixels = nonlinear_transformation(non_black_pixels)
        # # 确保变换后的像素值在0到255之间
        # transformed_pixels = np.clip(transformed_pixels, 0, 255)
        # # 将变换后的像素值赋值回输出图像
        # transformed_image[mask] = transformed_pixels
        # -------------------- 该部分注释结束 ----------------------------
        if self.apply_transform:
            transformed_image = apply_bezier_gray_transform(image)
        else:
            transformed_image = image
        #transformed_image = apply_bezier_gray_transform(image)
        # 将输出图像转换为uint8类型
        transformed_image = transformed_image.astype(np.uint8)
        # print(transformed_image.shape)
        # image = nonlinear_transformation(image)
        transformed_image = transformed_image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        # if label.max() > 1:
        #     label = label / 255

        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            transformed_image = self.augment(transformed_image, flipCode)
            label = self.augment(label, flipCode)
        return transformed_image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("MoE-test")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)

#  上面是脑数据集的加载方式
# --------------------------分割线---------------------------------------
#  下面是.nii.gz文件的加载方式

# import os
# import numpy as np
# import torch
# import cv2
# import glob
# from torch.utils.data import Dataset
# import random
# import nibabel as nib
# from bezier_curve_new import apply_bezier_gray_transform

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# class ISBI_Loader(Dataset):
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.nii.gz'))
#         self.samples = []  # 存储每个切片的病例路径和切片索引

#         # 遍历所有病例，记录每个切片的索引
#         for img_path in self.imgs_path:
#             # 读取病例以确定切片数量
#             img_nii = nib.load(img_path)
#             num_slices = img_nii.shape[-1]  # 假设切片在最后一个维度
#             for slice_idx in range(num_slices):
#                 self.samples.append((img_path, slice_idx))

#     def augment(self, image, flipCode):
#         # 处理单通道图像，输入形状为(1, H, W)
#         img_squeezed = image.squeeze(0)  # 转换为(H, W)
#         flipped = cv2.flip(img_squeezed, flipCode)
#         return flipped[None, :, :]  # 恢复为(1, H, W)

#     def __getitem__(self, index):
#         img_path, slice_idx = self.samples[index]

#         # 构建标签路径
#         base_name = os.path.basename(img_path)
#         case_name = os.path.splitext(base_name)[0].replace('.nii', '')  # 确保去除可能的.nii后缀
#         label_path = os.path.join(self.data_path, 'label', f"{case_name}_segmentation.nii.gz")

#         # 读取图像和标签的对应切片
#         image_nii = nib.load(img_path)
#         label_nii = nib.load(label_path)
#         image_slice = image_nii.get_fdata()[:, :, slice_idx].astype(np.uint8)
#         label_slice = label_nii.get_fdata()[:, :, slice_idx].astype(np.uint8)

#         # 应用非线性变换
#         transformed_image = apply_bezier_gray_transform(image_slice).astype(np.uint8)

#         # 调整形状为(1, H, W)
#         transformed_image = transformed_image[None, :, :]
#         label_slice = label_slice[None, :, :]

#         # 处理标签，将255转换为1
#         if label_slice.max() > 1:
#             label_slice = label_slice // 255  # 使用整数除法避免浮点问题

#         # 随机数据增强
#         flipCode = random.choice([-1, 0, 1, 2])
#         if flipCode != 2:
#             transformed_image = self.augment(transformed_image, flipCode)
#             label_slice = self.augment(label_slice, flipCode)

#         # 转换为张量
#         transformed_image = torch.from_numpy(transformed_image).float()
#         label_slice = torch.from_numpy(label_slice).float()

#         return transformed_image, label_slice

#     def __len__(self):
#         return len(self.samples)


