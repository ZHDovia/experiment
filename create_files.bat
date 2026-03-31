@echo off
chcp 65001 >nul
cd /d "D:\006科研训练\纯每周训练\week7"

echo 正在创建代码文件...

:: 创建 config.py
(
echo import os
echo.
echo class Config:
echo     BASE_DIR = r'D:\006科研训练\纯每周训练\week7'
echo     TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'DIV2K', 'train_HR')
echo     VAL_DIR = os.path.join(BASE_DIR, 'data', 'DIV2K', 'valid_HR')
echo     TEST_DIR = os.path.join(BASE_DIR, 'data', 'test', 'Set5')
echo     CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
echo     RESULT_DIR = os.path.join(BASE_DIR, 'data', 'test', 'result')
echo     SIGMA_TRAIN = 25
echo     PATCH_SIZE = 128
echo     BATCH_SIZE = 4
echo     EPOCHS = 10
echo     LR = 1e-4
echo     STEP_SIZE = 30
echo     GAMMA = 0.5
echo     GRAYSCALE = True
echo     TEST_SIGMAS = [15, 25, 35, 50]
echo.
echo     @classmethod
echo     def create_dirs(cls):
echo         os.makedirs(cls.TRAIN_DIR, exist_ok=True)
echo         os.makedirs(cls.VAL_DIR, exist_ok=True)
echo         os.makedirs(cls.TEST_DIR, exist_ok=True)
echo         os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
echo         os.makedirs(cls.RESULT_DIR, exist_ok=True)
) > config.py

:: 创建 models\unet.py
if not exist models mkdir models
(
echo import torch
echo import torch.nn as nn
echo import torch.nn.functional as F
echo.
echo class DoubleConv(nn.Module):
echo     def __init__(self, in_channels, out_channels):
echo         super(DoubleConv, self).__init__()
echo         self.conv = nn.Sequential(
echo             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
echo             nn.BatchNorm2d(out_channels),
echo             nn.ReLU(inplace=True),
echo             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
echo             nn.BatchNorm2d(out_channels),
echo             nn.ReLU(inplace=True)
echo         )
echo     def forward(self, x):
echo         return self.conv(x)
echo.
echo class UNet(nn.Module):
echo     def __init__(self, n_channels=1, n_classes=1):
echo         super(UNet, self).__init__()
echo         self.inc = DoubleConv(n_channels, 64)
echo         self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
echo         self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
echo         self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
echo         self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))
echo         self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
echo         self.up_conv1 = DoubleConv(512, 256)
echo         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
echo         self.up_conv2 = DoubleConv(256, 128)
echo         self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
echo         self.up_conv3 = DoubleConv(128, 64)
echo         self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
echo         self.up_conv4 = DoubleConv(128, 64)
echo         self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
echo.
echo     def forward(self, x):
echo         x1 = self.inc(x)
echo         x2 = self.down1(x1)
echo         x3 = self.down2(x2)
echo         x4 = self.down3(x3)
echo         x5 = self.down4(x4)
echo         x = self.up1(x5)
echo         x = torch.cat([x, x4], dim=1)
echo         x = self.up_conv1(x)
echo         x = self.up2(x)
echo         x = torch.cat([x, x3], dim=1)
echo         x = self.up_conv2(x)
echo         x = self.up3(x)
echo         x = torch.cat([x, x2], dim=1)
echo         x = self.up_conv3(x)
echo         x = self.up4(x)
echo         x = torch.cat([x, x1], dim=1)
echo         x = self.up_conv4(x)
echo         return self.outc(x)
) > models\unet.py

:: 创建空的 __init__.py
echo. > models\__init__.py
echo. > utils\__init__.py

:: 创建 train.py
(
echo import sys
echo import os
echo sys.path.insert(0, r'D:\006科研训练\纯每周训练\week7')
echo.
echo import torch
echo import torch.nn as nn
echo import torch.optim as optim
echo from torch.utils.data import Dataset, DataLoader
echo import numpy as np
echo import cv2
echo from tqdm import tqdm
echo from config import Config
echo from models.unet import UNet
echo.
echo class SimpleDataset(Dataset):
echo     def __init__(self, root_dir, sigma=25, patch_size=128, is_grayscale=True):
echo         self.image_paths = []
echo         if os.path.exists(root_dir):
echo             for f in os.listdir(root_dir):
echo                 if f.endswith(('.png', '.jpg')):
echo                     self.image_paths.append(os.path.join(root_dir, f))
echo         self.sigma = sigma
echo         self.patch_size = patch_size
echo         self.is_grayscale = is_grayscale
echo.
echo     def __len__(self):
echo         return len(self.image_paths) * 5
echo.
echo     def __getitem__(self, idx):
echo         img_idx = idx %% len(self.image_paths)
echo         img = cv2.imread(self.image_paths[img_idx])
echo         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
echo         if self.is_grayscale:
echo             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
echo             img = np.expand_dims(img, axis=-1)
echo         img = cv2.resize(img, (self.patch_size, self.patch_size))
echo         img = img.astype(np.float32) / 255.0
echo         noise = np.random.normal(0, self.sigma/255.0, img.shape)
echo         noisy = np.clip(img + noise, 0, 1)
echo         img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
echo         noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).float()
echo         return noisy_tensor, img_tensor
echo.
echo def train():
echo     Config.create_dirs()
echo     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
echo     print(f"使用设备: {device}")
echo.
echo     train_dataset = SimpleDataset(Config.TRAIN_DIR, sigma=Config.SIGMA_TRAIN, patch_size=Config.PATCH_SIZE)
echo     train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
echo.
echo     model = UNet().to(device)
echo     criterion = nn.MSELoss()
echo     optimizer = optim.Adam(model.parameters(), lr=Config.LR)
echo.
echo     print("开始训练...")
echo     for epoch in range(Config.EPOCHS):
echo         model.train()
echo         total_loss = 0
echo         for noisy, clean in tqdm(train_loader):
echo             noisy, clean = noisy.to(device), clean.to(device)
echo             optimizer.zero_grad()
echo             output = model(noisy)
echo             loss = criterion(output, clean)
echo             loss.backward()
echo             optimizer.step()
echo             total_loss += loss.item()
echo         print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
echo.
echo     torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, 'unet_grayscale.pth'))
echo     print("训练完成！")
echo.
echo if __name__ == '__main__':
echo     train()
) > train.py

echo 所有文件创建完成！
echo.
echo 现在请运行：
echo cd /d "D:\006科研训练\纯每周训练\week7"
echo unet_env\Scripts\activate
echo python train.py
pause