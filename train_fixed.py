# train_fixed.py - 修复保存路径问题
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 配置
PATCH_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
SIGMA_TRAIN = 25

# 确保 checkpoints 文件夹存在
os.makedirs('checkpoints', exist_ok=True)

class FakeDataset(Dataset):
    def __init__(self, num_samples=250, sigma=25, patch_size=128):
        self.num_samples = num_samples
        self.sigma = sigma
        self.patch_size = patch_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img = np.random.rand(self.patch_size, self.patch_size).astype(np.float32)
        img = np.expand_dims(img, axis=-1)
        noise = np.random.normal(0, self.sigma/255.0, img.shape)
        noisy = np.clip(img + noise, 0, 1)
        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        noisy_tensor = torch.from_numpy(noisy.copy()).permute(2, 0, 1).float()
        return noisy_tensor, img_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
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
        return self.outc(x)

def train():
    print("=" * 60)
    print("U-Net 图像去噪训练")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    train_dataset = FakeDataset(num_samples=250, sigma=SIGMA_TRAIN, patch_size=PATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"开始训练 {EPOCHS} 个epoch...")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for noisy, clean in loop:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, 平均损失: {avg_loss:.6f}")
    
    # 保存模型 - 确保路径存在
    model_path = 'checkpoints/unet_grayscale.pth'
    torch.save(model.state_dict(), model_path)
    print("-" * 60)
    print(f"训练完成！")
    print(f"模型已保存到: {model_path}")
    
    # 验证文件是否保存成功
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / 1024
        print(f"文件大小: {file_size:.1f} KB")
    else:
        print("错误: 模型保存失败！")

if __name__ == '__main__':
    train()
