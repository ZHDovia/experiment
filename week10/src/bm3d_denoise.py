# src/bm3d_denoise.py
import bm3d
import numpy as np

def bm3d_denoise(noisy_img, sigma):
    """
    BM3D 去噪
    noisy_img: 噪声图像，范围[0,1]
    sigma: 噪声水平 (15, 25, 35, 50)
    返回: 去噪图像，范围[0,1]
    """
    # bm3d 库要求输入范围 [0, 255]
    noisy_uint8 = np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8)
    
    # 调用 BM3D
    denoised_uint8 = bm3d.bm3d(noisy_uint8, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    
    # 转回 [0, 1] 范围
    denoised = denoised_uint8.astype(np.float32) / 255.0
    return denoised