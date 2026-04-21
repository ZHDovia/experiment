# src/self2self.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=48):
        super(SimpleUNet, self).__init__()
        self.dropout = nn.Dropout2d(p=0.3)
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, features, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features*2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(features*2, features*2, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*2, features*4, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(features*4, features*4, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(features*4 + features*2, features*2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(features*2, features*2, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(features*2 + features, features, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(features, features, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        self.output = nn.Conv2d(features, out_ch, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        b = self.dropout(b)
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.dropout(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.output(d1)


def self2self_denoise(noisy_img, sigma, device='cuda', epochs=100, ensemble=30):
    H, W = noisy_img.shape
    
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    noisy_padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    noisy_tensor = torch.from_numpy(noisy_padded).float().unsqueeze(0).unsqueeze(0).to(device)
    
    net = SimpleUNet(in_ch=1, out_ch=1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    
    p = 0.3
    
    net.train()
    for epoch in range(epochs):
        mask = torch.bernoulli(torch.full_like(noisy_tensor, p)).to(device)
        input_tensor = noisy_tensor * mask
        target = noisy_tensor * (1 - mask)
        
        optimizer.zero_grad()
        output = net(input_tensor)
        loss = mse_loss(output * (1 - mask), target)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    net.eval()
    predictions = []
    for _ in range(ensemble):
        mask = torch.bernoulli(torch.full_like(noisy_tensor, p)).to(device)
        with torch.no_grad():
            pred = net(noisy_tensor * mask)
        predictions.append(pred.cpu().numpy())
    
    denoised = np.median(predictions, axis=0).squeeze()
    
    if pad_h > 0 or pad_w > 0:
        denoised = denoised[:H, :W]
    
    return np.clip(denoised, 0, 1)