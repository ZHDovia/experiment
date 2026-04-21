# src/metrics.py
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def compute_psnr(clean, denoised):
    """计算PSNR，输入范围[0,1]"""
    return peak_signal_noise_ratio(clean, denoised, data_range=1.0)

def compute_ssim(clean, denoised):
    """计算SSIM，输入范围[0,1]"""
    return structural_similarity(clean, denoised, data_range=1.0)