# plot_results.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（Windows 用 SimHei，避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
DATA_PATH = './data/Set12'
OUTPUT_PATH = './outputs'

# 选择图像（01.png 或 05.png）
IMAGE_NAME = '05'
CLEAN_IMG = f'{DATA_PATH}/{IMAGE_NAME}.png'

# 噪声水平
SIGMAS = [15, 25, 35, 50]

# 方法配置：(方法名, 文件夹名, 显示名称)
METHODS = [
    ('bm3d', 'bm3d', 'BM3D'),
    ('n2n', 'n2n', 'N2N'),
    ('dip', 'dip', 'DIP'),
    ('self2self', 'self2self', 'Self2Self'),
]

def load_image(path, gray=True):
    """加载图像，返回 [0,1] 范围的 float32"""
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return np.zeros((256, 256), dtype=np.float32)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return img.astype(np.float32) / 255.0

def add_gaussian_noise(img, sigma):
    """添加高斯噪声"""
    noise = np.random.randn(*img.shape).astype(np.float32) * (sigma / 255.0)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def main():
    # 固定随机种子，使噪声可复现
    np.random.seed(42)
    
    # 加载原图
    clean = load_image(CLEAN_IMG)
    H, W = clean.shape
    
    # 创建大图：5列（噪声+4种方法）× 4行（4种噪声水平）
    fig1, axes1 = plt.subplots(4, 5, figsize=(16, 12))
    
    for row, sigma in enumerate(SIGMAS):
        # 生成噪声图
        noisy = add_gaussian_noise(clean, sigma)
        
        # 第一列：噪声图
        axes1[row, 0].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        axes1[row, 0].set_title(f'Noisy σ={sigma}', fontsize=10)
        axes1[row, 0].axis('off')
        
        # 后续列：各方法去噪结果
        for col, (method_key, folder, display_name) in enumerate(METHODS):
            denoised_path = f'{OUTPUT_PATH}/{folder}/{IMAGE_NAME}_sigma{sigma}.png'
            denoised = load_image(denoised_path)
            
            axes1[row, col+1].imshow(denoised, cmap='gray', vmin=0, vmax=1)
            axes1[row, col+1].set_title(f'{display_name}', fontsize=10)
            axes1[row, col+1].axis('off')
    
    # 添加行标签（最左侧）
    for row, sigma in enumerate(SIGMAS):
        axes1[row, 0].set_ylabel(f'σ={sigma}', fontsize=12, rotation=0, labelpad=30)
    
    plt.tight_layout()
    plt.savefig(f'./comparison_{IMAGE_NAME}_all.png', dpi=300, bbox_inches='tight')
    print(f"Saved: ./comparison_{IMAGE_NAME}_all.png")
    plt.close()
    
    # 创建第二张图：两张原图并排
    clean_01 = load_image(f'{DATA_PATH}/01.png')
    clean_05 = load_image(f'{DATA_PATH}/05.png')
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
    axes2[0].imshow(clean_01, cmap='gray', vmin=0, vmax=1)
    axes2[0].set_title('Ground Truth: 01.png', fontsize=12)
    axes2[0].axis('off')
    
    axes2[1].imshow(clean_05, cmap='gray', vmin=0, vmax=1)
    axes2[1].set_title('Ground Truth: 05.png', fontsize=12)
    axes2[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('./ground_truth_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: ./ground_truth_comparison.png")
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    main()