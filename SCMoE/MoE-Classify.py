from PIL import Image
import os
# 设置环境变量以避免某些库之间的冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 定义门控网络
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, input_shape, n_classes):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([UNet(input_shape[0], 1) for _ in range(num_experts)])  # 输出通道数为1
        self.gating_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        gating_outputs = torch.softmax(self.gating_network(x), dim=1)
        print(expert_outputs.shape)
        print(gating_outputs.unsqueeze(2).unsqueeze(3).shape)
        output = torch.sum(gating_outputs.unsqueeze(2).unsqueeze(3) * expert_outputs, dim=0)
        return torch.sigmoid(output)  # 直接返回sigmoid激活后的结果

# 定义数据集
class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        for phase in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = os.path.join(self.data_dir, phase, class_name)
                self.data.extend([os.path.join(class_dir, file) for file in os.listdir(class_dir)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  # 转换为3通道
        label = 1 if 'PNEUMONIA' in img_path else 0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)  # 确保标签是浮点数

# 定义main函数
def main():
    # 设置数据加载器
    data_dir = 'chest_xray'
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = ChestXRayDataset(data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 初始化模型和优化器
    num_experts = 4
    model = MixtureOfExperts(num_experts, (3, 256, 256), 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    # 训练专家模型
    print("现在开始训练专家模型")
    for expert in model.experts:
        expert_optimizer = optim.Adam(expert.parameters(), lr=0.001)
        expert_scheduler = optim.lr_scheduler.StepLR(expert_optimizer, step_size=1, gamma=0.7)
        for epoch in range(1):
            expert.train()
            for images, labels in tqdm(train_loader):
                expert_optimizer.zero_grad()
                outputs = expert(images)
                outputs = torch.sigmoid(outputs)
                labels = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                labels = labels.expand(-1, 1, 256, 256)
                loss = nn.BCELoss()(outputs, labels)
                loss.backward()
                expert_optimizer.step()
            expert_scheduler.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print("专家模型训练结束")

    # 训练门控网络
    model.train()
    print("现在开始训练门控网络")
    for epoch in range(1):
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            # 将标签扩展为 [batch_size, num_experts, height, width]
            # 首先将labels扩展到 [batch_size, 1, 1, 1]
            labels = labels.view(-1, 1, 1, 1)  # 现在形状是 [batch_size, 1, 1, 1]
            # 然后扩展到 [batch_size, num_experts, height, width]
            labels = labels.expand(-1, num_experts, 256, 256)  # 扩展到 [batch_size, num_experts, height, width]
            loss = nn.BCELoss()(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print("门控网络训练结束")

    # 评估模型
    print("现在开始进行模型评估")
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze()
            labels = labels.view(-1, 1, 1, 1)  # 现在形状是 [batch_size, 1, 1, 1]
            # 然后扩展到 [batch_size, num_experts, height, width]
            labels = labels.expand(-1, num_experts, 256, 256)  # 扩展到 [batch_size, num_experts, height, width]
            loss = nn.BCELoss()(outputs, labels)
            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total_samples += labels.numel()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total_samples
    print(f'Val Loss: {avg_loss:.4f}')
    print(f'Val Accuracy: {accuracy:.4f}')
    print("模型评估结束")

if __name__ == '__main__':
    main()