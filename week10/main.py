# main.py
import os
import sys
import csv
sys.path.append(os.path.dirname(__file__))

import torch
from src.data_loader import load_set12, add_gaussian_noise, save_image
from src.metrics import compute_psnr, compute_ssim

# ====== 选择要跑的方法 ======
# from src.n2n_denoise import n2n_denoise as denoise_func
# from src.dip import dip_denoise as denoise_func
from src.self2self import self2self_denoise as denoise_func

METHOD_NAME = "self2self"  # 改成对应名字：n2n / dip / self2self

# ====== 配置 ======
TARGET_IMAGES = ['01.png', '05.png']
SIGMAS = [15, 25, 35, 50]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_benchmark():
    os.makedirs(f'./outputs/{METHOD_NAME}/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    
    images = load_set12('./data/Set12/')
    
    # 计算总任务数
    total_tasks = len(TARGET_IMAGES) * len(SIGMAS)
    current_task = 0
    
    csv_path = f'./logs/{METHOD_NAME}_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'Sigma', 'PSNR', 'SSIM'])
        
        for name, clean in images:
            if name not in TARGET_IMAGES:
                continue
            
            for sigma in SIGMAS:
                current_task += 1
                print(f"\n{'='*50}")
                print(f"[{current_task}/{total_tasks}] Method: {METHOD_NAME} | Image: {name} | σ={sigma}")
                print(f"Device: {DEVICE}")
                print('='*50)
                
                noisy = add_gaussian_noise(clean, sigma)
                denoised = denoise_func(noisy, sigma, device=DEVICE)
                
                psnr = compute_psnr(clean, denoised)
                ssim = compute_ssim(clean, denoised)
                
                save_image(denoised, f'./outputs/{METHOD_NAME}/{name[:-4]}_sigma{sigma}.png')
                writer.writerow([name, sigma, f"{psnr:.2f}", f"{ssim:.4f}"])
                
                print(f"Done: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
    
    print(f"\n[METHOD_NAME] benchmark completed! Results saved to {csv_path}")

if __name__ == "__main__":
    run_benchmark()