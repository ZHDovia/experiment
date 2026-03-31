# train_div2k.py - DIV2K完整数据集训练
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 配置
PATCH_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
SIGMA_TRAIN = 25

os.makedirs('checkpoints', exist_ok=True)

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, sigma=25, patch_size=128):
        self.image_paths = []
        if os.path.exists(root_dir):
            for f in os.listdir(root_dir):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, f))
        print(f"在 {root_dir} 中找到 {len(self.image_paths)} 张图像")
        self.sigma = sigma
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.image_paths) * 5
    
    def __getitem__(self, idx):
        img_idx = idx % len(self.image_paths)
        img = cv2.imread(self.image_paths[img_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        h, w = img.shape
        if h >= self.patch_size and w >= self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            img = img[top:top+self.patch_size, left:left+self.patch_size]
        else:
            img = cv2.resize(img, (self.patch_size, self.patch_size))
        
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        
        noise = np.random.normal(0, self.sigma/255.0, img.shape)
        noisy = np.clip(img + noise, 0, 1)
        
        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        noisy_tensor = torch.from_numpy(noisy.copy()).permute(2, 0, 1).float()
        
        return noisy_tensor, img_tensor

# 从 train_fixed.py 导入 UNet 模型
from train_fixed import UNet

def train():
    print("=" * 60)
    print("DIV2K 完整数据集训练 (σ=25)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查数据集
    train_dir = 'data/DIV2K/train_HR'
    val_dir = 'data/DIV2K/valid_HR'
    
    if not os.path.exists(train_dir):
        print(f"错误: 训练集不存在: {train_dir}")
        return
    if not os.path.exists(val_dir):
        print(f"错误: 验证集不存在: {val_dir}")
        return
    
    train_dataset = DIV2KDataset(train_dir, sigma=SIGMA_TRAIN, patch_size=PATCH_SIZE)
    val_dataset = DIV2KDataset(val_dir, sigma=SIGMA_TRAIN, patch_size=PATCH_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"开始训练 {EPOCHS} 个epoch...")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [训练]')
        
        for noisy, clean in loop:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [验证]'):
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/unet_div2k.pth')
            print(f"  -> 保存最佳模型，Val Loss={best_loss:.6f}")
    
    print("-" * 60)
    print(f"训练完成！最佳验证损失: {best_loss:.6f}")
    print(f"模型已保存: checkpoints/unet_div2k.pth")

if __name__ == '__main__':
    train()