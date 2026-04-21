# src/n2n_denoise.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.append(os.path.dirname(__file__))
from arch_unet import UNet

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size, w // block_size)

def generate_mask_pair(img, device):
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,), dtype=torch.bool, device=device)
    
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64, device=device)
    
    rd_idx = torch.randint(low=0, high=8, size=(n * h // 2 * w // 2,), device=device)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0, end=n * h // 2 * w // 2 * 4, step=4,
                                dtype=torch.int64, device=device).reshape(-1, 1)
    
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n, c, h // 2, w // 2, dtype=img.dtype, device=img.device)
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i+1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i+1, :, :] = img_per_channel[mask].reshape(n, h//2, w//2, 1).permute(0, 3, 1, 2)
    return subimage

def train_n2n_single(noisy_img, sigma, device='cuda', epochs=100):
    H, W = noisy_img.shape
    noisy_img = noisy_img.astype(np.float32)
    img_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(device)
    
    patch_size = 256
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    _, _, H_pad, W_pad = img_tensor.shape
    
    net = UNet(in_nc=1, out_nc=1, n_feature=48, blindspot=False).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    net.train()
    for epoch in range(epochs):
        h_start = np.random.randint(0, max(1, H_pad - patch_size + 1))
        w_start = np.random.randint(0, max(1, W_pad - patch_size + 1))
        patch = img_tensor[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size]
        
        mask1, mask2 = generate_mask_pair(patch, device)
        sub1 = generate_subimages(patch, mask1)
        sub2 = generate_subimages(patch, mask2)
        
        with torch.no_grad():
            denoised_full = net(patch)
        denoised_sub1 = generate_subimages(denoised_full, mask1)
        denoised_sub2 = generate_subimages(denoised_full, mask2)
        
        output = net(sub1)
        
        Lambda = epoch / epochs * 2.0
        diff = output - sub2
        exp_diff = denoised_sub1 - denoised_sub2
        loss1 = torch.mean(diff ** 2)
        loss2 = Lambda * torch.mean((diff - exp_diff) ** 2)
        loss = loss1 + loss2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ====== 添加进度显示 ======
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    net.eval()
    with torch.no_grad():
        denoised = net(img_tensor).squeeze().cpu().numpy()
    
    if pad_h > 0 or pad_w > 0:
        denoised = denoised[:H, :W]
    
    return np.clip(denoised, 0, 1)

def n2n_denoise(noisy_img, sigma, device='cuda'):
    return train_n2n_single(noisy_img, sigma, device=device, epochs=100)