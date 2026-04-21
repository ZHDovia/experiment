# src/data_loader.py
import os
import cv2
import numpy as np

def load_set12(data_path='./data/Set12/'):
    """
    加载Set12所有图像
    返回: list of (filename, clean_image)
    """
    images = []
    
    for i in range(1, 13):
        fname = f"{i:02d}.png"
        filepath = os.path.join(data_path, fname)
        
        if os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img.astype(np.float32) / 255.0
                images.append((fname, img))
        else:
            print(f"Warning: {filepath} not found")
    
    print(f"Loaded {len(images)} images from {data_path}")
    return images

def add_gaussian_noise(img, sigma):
    """
    添加高斯噪声
    img: 干净图像，范围[0,1]
    sigma: 噪声水平 (15, 25, 35, 50)
    返回: 噪声图像，范围[0,1]
    """
    noise = np.random.randn(*img.shape).astype(np.float32) * (sigma / 255.0)
    noisy = img + noise
    return np.clip(noisy, 0.0, 1.0)

def save_image(img, filepath):
    """保存图像，输入范围[0,1]"""
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(filepath, img_uint8)