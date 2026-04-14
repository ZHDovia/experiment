
import torch
import numpy as np
import cv2
import pandas as pd
import os
from train_fixed import UNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_noise(img, sigma):
    noise = np.random.normal(0, sigma/255.0, img.shape)
    return np.clip(img + noise, 0, 1)

def median_denoise(noisy, sigma):
    from scipy.ndimage import median_filter
    size = int(sigma / 10) * 2 + 1
    return median_filter(noisy, size=max(3, min(size, 7)))

def dncnn_denoise(noisy, model):
    tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return model(tensor).squeeze().cpu().numpy().clip(0, 1)

def n2v_denoise(noisy, model):
    tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)
    shifted = torch.roll(tensor, shifts=(1, 1), dims=(2, 3))
    with torch.no_grad():
        pred = model(shifted)
        return torch.roll(pred, shifts=(-1, -1), dims=(2, 3)).squeeze().cpu().numpy().clip(0, 1)

def main():
    print("=" * 60)
    print("四种去噪方法对比实验")
    print("=" * 60)
    
    # 加载模型
    model_dncnn = UNet().to(device)
    model_dncnn.load_state_dict(torch.load('checkpoints/unet_div2k.pth', map_location=device))
    model_dncnn.eval()
    
    model_n2n = UNet().to(device)
    model_n2n.load_state_dict(torch.load('checkpoints/n2n_final.pth', map_location=device))
    model_n2n.eval()
    
    print("✅ 模型加载完成\n")
    
    images = ['baboon', 'ppt3']
    sigmas = [15, 25, 35, 50]
    results = []
    
    for img_name in images:
        img_path = f'data/test/set14/{img_name}.png'
        original = cv2.imread(img_path, 0) / 255.0
        
        for sigma in sigmas:
            np.random.seed(42)
            noisy = add_noise(original, sigma)
            
            # Median
            denoised = median_denoise(noisy, sigma)
            results.append(['Median', img_name, sigma, 
                          psnr(original, denoised, data_range=1),
                          ssim(original, denoised, data_range=1)])
            
            # DnCNN
            denoised = dncnn_denoise(noisy, model_dncnn)
            results.append(['DnCNN', img_name, sigma,
                          psnr(original, denoised, data_range=1),
                          ssim(original, denoised, data_range=1)])
            
            # Noise2Void
            denoised = n2v_denoise(noisy, model_dncnn)
            results.append(['Noise2Void', img_name, sigma,
                          psnr(original, denoised, data_range=1),
                          ssim(original, denoised, data_range=1)])
            
            # Neighbor2Neighbor
            denoised = dncnn_denoise(noisy, model_n2n)
            results.append(['Neighbor2Neighbor', img_name, sigma,
                          psnr(original, denoised, data_range=1),
                          ssim(original, denoised, data_range=1)])
    
    # 生成表格
    df = pd.DataFrame(results, columns=['Method', 'Image', 'Sigma', 'PSNR', 'SSIM'])
    pivot = df.pivot_table(index='Method', columns=['Image', 'Sigma'], values='PSNR').round(2)
    
    print("\nPSNR对比表:")
    print("=" * 60)
    print(pivot)
    
    df.to_csv('comparison_results.csv', index=False)
    print("\n✅ 结果已保存到 comparison_results.csv")

if __name__ == '__main__':
    main()