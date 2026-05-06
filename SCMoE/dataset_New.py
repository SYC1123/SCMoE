import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ISBI_Loader_New(Dataset):
    def __init__(self, data_path, return_domain=False):
        """
        支持两种结构：
        1. data_path/
              DomainA/
                  image/...
                  label/...
              DomainB/
                  image/...
                  label/...
        2. data_path/
              image/Case01/
              label/Case01/
        """
        self.return_domain = return_domain
        self.samples = []

        # 判断是否是“多域模式”（每个子文件夹是一个域）
        subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        is_multi_domain = any(os.path.isdir(os.path.join(data_path, d, 'image')) for d in subdirs)

        if is_multi_domain:
            self.domain_names = sorted([d for d in subdirs if os.path.isdir(os.path.join(data_path, d, 'image'))])
            self.dom2id = {d: i for i, d in enumerate(self.domain_names)}

            for dom in self.domain_names:
                img_pattern = os.path.join(data_path, dom, 'image', '**', '*.png')
                for p in glob.glob(img_pattern, recursive=True):
                    self.samples.append((p, self.dom2id[dom]))
        else:
            # 单域模式：image 和 label 在 data_path 下
            self.domain_names = ['default']
            self.dom2id = {'default': 0}
            img_pattern = os.path.join(data_path, 'image', '**', '*.png')
            for p in glob.glob(img_pattern, recursive=True):
                self.samples.append((p, 0))

    def augment(self, arr, flipCode):
        return cv2.flip(arr, flipCode)

    def __getitem__(self, idx):
        img_path, dom_id = self.samples[idx]
        lbl_path = img_path.replace(os.sep + 'image' + os.sep, os.sep + 'label' + os.sep)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        img = img[np.newaxis] / 255.0
        lbl = lbl[np.newaxis]
        if lbl.max() > 1:
            lbl = lbl // 255.0

        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            img = self.augment(img, flipCode)
            lbl = self.augment(lbl, flipCode)

        img_t = torch.from_numpy(img).float()
        lbl_t = torch.from_numpy(lbl).float()

        if self.return_domain:
            dom_t = torch.tensor(dom_id, dtype=torch.long)
            return img_t, lbl_t, dom_t
        else:
            return img_t, lbl_t

    def __len__(self):
        return len(self.samples)

    
if __name__ == "__main__":
    # 这里打开 return_domain
    isbi_dataset = ISBI_Loader_New(
        data_path="MoE-test",
        return_domain=True
    )
    print("总样本数：", len(isbi_dataset))
    print("发现域（subfolder）数：", len(isbi_dataset.domain_names))
    train_loader = torch.utils.data.DataLoader(
        dataset=isbi_dataset,
        batch_size=2,
        shuffle=True
    )

    for image, label, domain_ids in train_loader:
        # domain_ids 是一个长度为 batch_size 的 LongTensor
        print("image.shape =", image.shape)
        print("label.shape =", label.shape)
        print("domain_ids.shape =", domain_ids.shape)     # torch.Size([2])
        print("domain_ids =", domain_ids)        # e.g. [0,2]
        break
