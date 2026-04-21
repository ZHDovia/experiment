# src/dip.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DIPNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=1, num_features=128):
        super(DIPNet, self).__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.1)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_features, num_features*2, 3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_features*2, num_features*2, 3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.LeakyReLU(0.1)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(num_features*2, num_features*4, 3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_features*4, num_features*4, 3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.LeakyReLU(0.1)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_features*4, num_features*8, 3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_features*8, num_features*8, 3, padding=1),
            nn.BatchNorm2d(num_features*8),
            nn.LeakyReLU(0.1)
        )
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(num_features*8 + num_features*4, num_features*4, 3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_features*4, num_features*4, 3, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.LeakyReLU(0.1)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(num_features*4 + num_features*2, num_features*2, 3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_features*2, num_features*2, 3, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.LeakyReLU(0.1)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_features*2 + num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.1)
        )
        
        self.output = nn.Conv2d(num_features, out_channels, 1)
        
    def forward(self, z):
        e1 = self.enc1(z)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.output(d1)
        return out

def dip_denoise(noisy_img, sigma, device='cuda'):
    H, W = noisy_img.shape
    
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    noisy_padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode='reflect')
    H_pad, W_pad = noisy_padded.shape
    
    noisy_tensor = torch.from_numpy(noisy_padded).float().unsqueeze(0).unsqueeze(0).to(device)
    
    torch.manual_seed(42)
    z = torch.randn(1, 32, H_pad, W_pad).to(device)
    
    net = DIPNet(in_channels=32, out_channels=1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    iter_map = {15: 1000, 25: 1200, 35: 1400, 50: 1500}
    num_iter = iter_map.get(sigma, 1200)
    
    net.train()
    for i in range(num_iter):
        optimizer.zero_grad()
        output = net(z)
        loss = mse_loss(output, noisy_tensor)
        loss.backward()
        optimizer.step()
        
        # ====== 添加进度显示 ======
        if (i + 1) % 100 == 0:
            print(f"    Iter {i+1}/{num_iter}, Loss: {loss.item():.6f}")
    
    net.eval()
    with torch.no_grad():
        denoised = net(z).squeeze().cpu().numpy()
    
    if pad_h > 0 or pad_w > 0:
        denoised = denoised[:H, :W]
    
    return np.clip(denoised, 0, 1)