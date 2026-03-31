# generate_comparison.py - 手动生成对比图
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_fixed import UNet

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('checkpoints/unet_div2k.pth', map_location=device))
model.eval()

def add_noise(image, sigma):
    noise = np.random.normal(0, sigma/255.0, image.shape)
    return np.clip(image + noise, 0, 1)

def denoise(model, noisy):
    noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        denoised = model(noisy_tensor)
    return denoised.squeeze(0).cpu().permute(1, 2, 0).numpy()

# 选择两张测试图像：baboon 和 ppt3
test_dir = 'data/test/Set14'
image_names = ['baboon', 'ppt3']
sigma = 25  # 使用训练噪声强度

for name in image_names:
    img_path = os.path.join(test_dir, f'{name}.png')
    if not os.path.exists(img_path):
        print(f"找不到: {img_path}")
        continue
    
    # 读取并转为灰度
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    
    # 添加噪声并去噪
    noisy = add_noise(img, sigma)
    denoised = denoise(model, noisy)
    
    # 保存对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.squeeze(), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(noisy.squeeze(), cmap='gray')
    axes[1].set_title(f'Noisy (σ={sigma})')
    axes[1].axis('off')
    
    axes[2].imshow(denoised.squeeze(), cmap='gray')
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs('data/test/official_results', exist_ok=True)
    plt.savefig(f'data/test/official_results/{name}_sigma{sigma}.png', dpi=150)
    plt.close()
    print(f"已保存: data/test/official_results/{name}_sigma{sigma}.png")

print("对比图生成完成！")