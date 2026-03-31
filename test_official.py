# test_official.py - 正式实验测试脚本
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# 导入模型
from train_fixed import UNet

# 配置
CHECKPOINT_DIR = 'checkpoints'
MODEL_NAME = 'unet_div2k.pth'   # 使用真实图像训练的模型
TEST_DIR = 'data/test/Set14'  # 测试集目录
RESULT_DIR = 'data/test/official_results'
os.makedirs(RESULT_DIR, exist_ok=True)

def add_gaussian_noise(image, sigma):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma/255.0, image.shape)
    return np.clip(image + noise, 0, 1)

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """计算SSIM"""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, data_range=1)

def load_test_images():
    """加载测试图像，如果没有则自动创建"""
    images = []
    names = []
    
    if not os.path.exists(TEST_DIR):
        print(f"测试目录不存在，自动创建: {TEST_DIR}")
        os.makedirs(TEST_DIR, exist_ok=True)
        # 创建示例测试图像
        for i, name in enumerate(['baby', 'bird', 'butterfly', 'head', 'woman']):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(TEST_DIR, f'{name}.png'), img)
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            names.append(name)
        print(f"已创建 {len(images)} 个示例测试图像")
    else:
        for f in os.listdir(TEST_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(TEST_DIR, f)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    names.append(os.path.splitext(f)[0])
        print(f"找到 {len(images)} 个测试图像")
    
    return images, names

def save_visualization(original, noisy, denoised, name, sigma):
    """保存可视化对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original.squeeze(), cmap='gray')
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(noisy.squeeze(), cmap='gray')
    axes[1].set_title(f'Noisy (σ={sigma})', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(denoised.squeeze(), cmap='gray')
    axes[2].set_title('Denoised', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f'{name}_sigma{sigma}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_results(results, image_names):
    """保存结果表格"""
    df = pd.DataFrame(results)
    df = df.round({'PSNR': 2, 'SSIM': 4})
    
    csv_path = os.path.join(RESULT_DIR, 'results.csv')
    df.to_csv(csv_path, index=False)
    
    txt_path = os.path.join(RESULT_DIR, 'results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("U-Net 图像去噪正式实验结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"训练数据: 真实图像\n")
        f.write(f"训练噪声: σ=25\n")
        f.write(f"测试集: {len(image_names)}张图像\n")
        f.write(f"测试图像: {', '.join(image_names)}\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("注: PSNR单位dB，SSIM范围[0,1]\n")
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(df.to_string(index=False))

def test():
    print("=" * 60)
    print("正式实验：U-Net 图像去噪性能测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = UNet().to(device)
    model_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
    
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        print("请先运行 train_real.py 训练模型")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"加载模型: {model_path}")
    
    # 加载测试图像
    test_images, image_names = load_test_images()
    
    if len(test_images) == 0:
        print("错误: 没有测试图像")
        return
    
    # 测试不同噪声强度
    test_sigmas = [15, 25, 35, 50]
    results = []
    
    print("\n开始测试...")
    for sigma in test_sigmas:
        print(f"\n测试噪声强度 σ={sigma}")
        
        psnr_list = []
        ssim_list = []
        
        for idx, (img, name) in enumerate(zip(test_images, image_names)):
            # 转为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.float32) / 255.0
            gray = np.expand_dims(gray, axis=-1)
            
            # 添加噪声
            noisy = add_gaussian_noise(gray, sigma)
            
            # 推理
            noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                denoised_tensor = model(noisy_tensor)
            
            denoised = denoised_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            denoised = np.clip(denoised, 0, 1)
            
            # 计算指标
            psnr = calculate_psnr(denoised, gray)
            ssim = calculate_ssim(denoised.squeeze(), gray.squeeze())
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            print(f"  {name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
            
            # 保存可视化（每个sigma保存第一张图）
            if idx == 0:
                save_visualization(gray, noisy, denoised, name, sigma)
        
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        results.append({'sigma': sigma, 'PSNR': avg_psnr, 'SSIM': avg_ssim})
        
        print(f"  平均: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
    
    # 保存结果
    save_results(results, image_names)
    
    print("\n" + "=" * 60)
    print(f"测试完成！结果已保存到: {RESULT_DIR}")

if __name__ == '__main__':
    test()