import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, metrics
from skimage.restoration import denoise_tv_chambolle
from skimage.util import random_noise
import bm3d
import os
import time

def add_gaussian_noise(img, sigma=0.1):
    return random_noise(img, mode='gaussian', var=sigma**2)

def prox_tv_high_quality(x, lambd, n_iter=50):
    return denoise_tv_chambolle(x, weight=lambd, eps=1e-5, max_num_iter=n_iter)

def ista_denoising(noisy, original, lambd=0.06, time_budget=10):
    x = noisy.copy()
    psnr_history = []
    time_history = []
    step_size = 0.05
    
    start_time = time.time()
    iter_count = 0
    
    while time.time() - start_time < time_budget:
        gradient = x - noisy
        x = x - step_size * gradient
        x = prox_tv_high_quality(x, lambd * step_size, n_iter=30)
        
        x_clipped = np.clip(x, 0, 1)
        psnr = metrics.peak_signal_noise_ratio(original, x_clipped)
        psnr_history.append(psnr)
        time_history.append(time.time() - start_time)
        iter_count += 1
        
        if iter_count % 100 == 0:
            print(f"   ISTA 迭代 {iter_count}, PSNR: {psnr:.2f} dB")

    print(f"   ISTA 完成! 总迭代: {iter_count}, 最终PSNR: {psnr_history[-1]:.2f} dB")
    return np.clip(x, 0, 1), psnr_history, time_history

def fista_denoising(noisy, original, lambd=0.08, time_budget=10):
    x = noisy.copy()
    y = x.copy()
    t = 1.0
    psnr_history = []
    time_history = []
    step_size = 0.05
    
    start_time = time.time()
    iter_count = 0
    
    while time.time() - start_time < time_budget:
        x_old = x.copy()
        
        gradient = y - noisy
        x = y - step_size * gradient
        x = prox_tv_high_quality(x, lambd * step_size, n_iter=30)
        
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_old)
        t = t_next
        
        x_clipped = np.clip(x, 0, 1)
        psnr = metrics.peak_signal_noise_ratio(original, x_clipped)
        psnr_history.append(psnr)
        time_history.append(time.time() - start_time)
        iter_count += 1
        
        if iter_count % 100 == 0:
            print(f"   FISTA 迭代 {iter_count}, PSNR: {psnr:.2f} dB")

    print(f"   FISTA 完成! 总迭代: {iter_count}, 最终PSNR: {psnr_history[-1]:.2f} dB")
    return np.clip(x, 0, 1), psnr_history, time_history

def admm_denoising(noisy, original, lambd=0.08, rho=0.5, time_budget=10):
    h, w = noisy.shape
    y = noisy.copy()
    
    x = denoise_tv_chambolle(y, weight=0.05, eps=1e-3, max_num_iter=20)
    z = np.zeros((2, h, w))
    u = np.zeros_like(z)
    
    psnr_history = []
    time_history = []
    
    def Dx(x):
        dx = np.zeros_like(x)
        dx[:, 1:] = x[:, 1:] - x[:, :-1]
        return dx
    
    def Dy(x):
        dy = np.zeros_like(x)
        dy[1:, :] = x[1:, :] - x[:-1, :]
        return dy
    
    def Dxt(dx):
        dxt = np.zeros_like(dx)
        dxt[:, 0] = -dx[:, 0]
        dxt[:, 1:-1] = -dx[:, 1:-1] + dx[:, :-2]
        dxt[:, -1] = dx[:, -2]
        return dxt
    
    def Dyt(dy):
        dyt = np.zeros_like(dy)
        dyt[0, :] = -dy[0, :]
        dyt[1:-1, :] = -dy[1:-1, :] + dy[:-2, :]
        dyt[-1, :] = dy[-2, :]
        return dyt
    
    fx = np.fft.fftfreq(w).reshape(1, -1)
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    DtD_freq = (2 - 2*np.cos(2*np.pi*fx)) + (2 - 2*np.cos(2*np.pi*fy))
    DtD_freq = DtD_freq + 1e-8
    
    def solve_x(y, rho, z, u):
        zx, zy = z
        ux, uy = u
        b = y + rho * (Dxt(zx - ux) + Dyt(zy - uy))
        b_freq = np.fft.fft2(b)
        x_freq = b_freq / (1 + rho * DtD_freq)
        return np.real(np.fft.ifft2(x_freq))
    
    def soft_threshold(v, threshold):
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
    
    start_time = time.time()
    iter_count = 0
    best_psnr = 0
    best_x = x.copy()
    
    rho_orig = rho
    
    while time.time() - start_time < time_budget:

        x_old = x.copy()
        z_old = z.copy()
        
    
        x = solve_x(y, rho, z, u)
        
   
        Dx_x = Dx(x)
        Dy_x = Dy(x)
        z_new_x = soft_threshold(Dx_x + u[0], lambd / rho)
        z_new_y = soft_threshold(Dy_x + u[1], lambd / rho)
        z = np.array([z_new_x, z_new_y])
        
        
        u[0] = u[0] + Dx_x - z[0]
        u[1] = u[1] + Dy_x - z[1]
        
        
        x_clipped = np.clip(x, 0, 1)
        psnr = metrics.peak_signal_noise_ratio(original, x_clipped)
        psnr_history.append(psnr)
        time_history.append(time.time() - start_time)
        iter_count += 1
        
    
        if psnr > best_psnr:
            best_psnr = psnr
            best_x = x.copy()
        
        
        if iter_count % 200 == 0:
            print(f"   ADMM 迭代 {iter_count}, PSNR: {psnr:.2f} dB, 时间: {time_history[-1]:.2f}秒")
        
        if iter_count % 20 == 0 and iter_count > 20:

            r_norm = np.linalg.norm(Dx_x - z[0]) + np.linalg.norm(Dy_x - z[1])
            s_norm = np.linalg.norm(rho * (z - z_old))
            
            if r_norm > 10 * s_norm:
                rho = min(rho * 1.5, 5.0)
            elif s_norm > 10 * r_norm:
                rho = max(rho / 1.5, 0.1)
    
    print(f"   ADMM 完成! 总迭代: {iter_count}, 最佳PSNR: {best_psnr:.2f} dB")
    return np.clip(best_x, 0, 1), psnr_history, time_history

def bm3d_denoising(noisy, original, sigma=0.1):
    start_time = time.time()
    denoised = bm3d.bm3d(noisy, sigma_psd=sigma)
    elapsed = time.time() - start_time
    return np.clip(denoised, 0, 1), elapsed

def load_test_image():
    from skimage import data
    img = img_as_float(data.camera())
    return img, "camera"

def plot_results(original, noisy, results_dict, times_dict, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('(a) Original', fontsize=12)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(noisy, cmap='gray')
    psnr_noisy = metrics.peak_signal_noise_ratio(original, noisy)
    axes[0,1].set_title(f'(b) Noisy\nPSNR: {psnr_noisy:.2f} dB', fontsize=12)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(results_dict['BM3D'], cmap='gray')
    psnr = metrics.peak_signal_noise_ratio(original, results_dict['BM3D'])
    ssim = metrics.structural_similarity(original, results_dict['BM3D'], data_range=1.0)
    axes[0,2].set_title(f'(c) BM3D\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}\nTime: {times_dict["BM3D"]:.2f}s', fontsize=10)
    axes[0,2].axis('off')
    
    axes[1,0].imshow(results_dict['ISTA-TV'], cmap='gray')
    psnr = metrics.peak_signal_noise_ratio(original, results_dict['ISTA-TV'])
    ssim = metrics.structural_similarity(original, results_dict['ISTA-TV'], data_range=1.0)
    axes[1,0].set_title(f'(d) ISTA-TV\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}\nTime: {times_dict["ISTA-TV"]:.2f}s\nIter: {times_dict["ISTA-iter"]}', fontsize=10)
    axes[1,0].axis('off')
    
    axes[1,1].imshow(results_dict['FISTA-TV'], cmap='gray')
    psnr = metrics.peak_signal_noise_ratio(original, results_dict['FISTA-TV'])
    ssim = metrics.structural_similarity(original, results_dict['FISTA-TV'], data_range=1.0)
    axes[1,1].set_title(f'(e) FISTA-TV\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}\nTime: {times_dict["FISTA-TV"]:.2f}s\nIter: {times_dict["FISTA-iter"]}', fontsize=10)
    axes[1,1].axis('off')
    
    axes[1,2].imshow(results_dict['ADMM-TV'], cmap='gray')
    psnr = metrics.peak_signal_noise_ratio(original, results_dict['ADMM-TV'])
    ssim = metrics.structural_similarity(original, results_dict['ADMM-TV'], data_range=1.0)
    axes[1,2].set_title(f'(f) ADMM-TV\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}\nTime: {times_dict["ADMM-TV"]:.2f}s\nIter: {times_dict["ADMM-iter"]}', fontsize=10)
    axes[1,2].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence_vs_time(histories_time, bm3d_psnr, save_path=None):
    plt.figure(figsize=(12, 8))
    
    colors = {'ISTA-TV': 'red', 'FISTA-TV': 'blue', 'ADMM-TV': 'green'}
    
    for algo_name in ['ISTA-TV', 'FISTA-TV', 'ADMM-TV']:
        if algo_name in histories_time:
            psnr_hist, time_hist = histories_time[algo_name]
            plt.plot(time_hist, psnr_hist, color=colors[algo_name], linewidth=2, 
                    label=f'{algo_name} (final: {psnr_hist[-1]:.2f} dB)')
    
    plt.axhline(y=bm3d_psnr, color='purple', linestyle='--', linewidth=2, 
                label=f'BM3D (final: {bm3d_psnr:.2f} dB)')
    
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title('PSNR Convergence vs Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim([20, 32])
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    result_folder = "denoising_results"
    os.makedirs(result_folder, exist_ok=True)
    print(f"结果保存至: {os.path.abspath(result_folder)}")
    
    img, img_name = load_test_image()
    print(f"图像: {img_name}, 尺寸: {img.shape}")
    
    sigma = 0.1
    noisy = add_gaussian_noise(img, sigma=sigma)
    print(f"噪声水平: sigma={sigma}")
    
    TIME_BUDGET = 10
    
    print("\n" + "="*70)
    print(f"开始去噪处理 - 统一时间预算: {TIME_BUDGET}秒")
    print("="*70)
    
    results = {}
    times = {}
    histories_time = {}
    
    print("\n[1] ISTA-TV (λ=0.06)...")
    res_ista, psnr_ista_hist, time_ista_hist = ista_denoising(noisy, img, lambd=0.06, time_budget=TIME_BUDGET)
    results['ISTA-TV'] = res_ista
    times['ISTA-TV'] = time_ista_hist[-1]
    times['ISTA-iter'] = len(psnr_ista_hist)
    histories_time['ISTA-TV'] = (psnr_ista_hist, time_ista_hist)
    
    print("\n[2] FISTA-TV (λ=0.08)...")
    res_fista, psnr_fista_hist, time_fista_hist = fista_denoising(noisy, img, lambd=0.08, time_budget=TIME_BUDGET)
    results['FISTA-TV'] = res_fista
    times['FISTA-TV'] = time_fista_hist[-1]
    times['FISTA-iter'] = len(psnr_fista_hist)
    histories_time['FISTA-TV'] = (psnr_fista_hist, time_fista_hist)
    
    print("\n[3] ADMM-TV (λ=0.08, ρ=0.5)...")
    res_admm, psnr_admm_hist, time_admm_hist = admm_denoising(noisy, img, lambd=0.08, rho=0.5, time_budget=TIME_BUDGET)
    results['ADMM-TV'] = res_admm
    times['ADMM-TV'] = time_admm_hist[-1]
    times['ADMM-iter'] = len(psnr_admm_hist)
    histories_time['ADMM-TV'] = (psnr_admm_hist, time_admm_hist)
    
    print("\n[4] BM3D...")
    res_bm3d, time_bm3d = bm3d_denoising(noisy, img, sigma=sigma)
    results['BM3D'] = res_bm3d
    times['BM3D'] = time_bm3d
    psnr_bm3d = metrics.peak_signal_noise_ratio(img, res_bm3d)
    print(f"   BM3D 完成! PSNR: {psnr_bm3d:.2f} dB, 用时: {time_bm3d:.2f}秒")
    
    print("\n" + "="*70)
    print("计算SSIM指标")
    print("="*70)
    ssim_values = {}
    for algo_name in results.keys():
        ssim = metrics.structural_similarity(img, results[algo_name], data_range=1.0)
        ssim_values[algo_name] = ssim
        print(f"   {algo_name}: SSIM = {ssim:.4f}")
    
    print("\n生成结果图表...")
    plot_results(img, noisy, results, times, 
                save_path=os.path.join(result_folder, 'figure1_results.png'))
    
    plot_convergence_vs_time(histories_time, psnr_bm3d,
                            save_path=os.path.join(result_folder, 'figure2_convergence_vs_time.png'))
    
    psnr_ista = metrics.peak_signal_noise_ratio(img, res_ista)
    psnr_fista = metrics.peak_signal_noise_ratio(img, res_fista)
    psnr_admm = metrics.peak_signal_noise_ratio(img, res_admm)

    print("="*90)
    print(f"{'算法':<15} {'PSNR(dB)':<15} {'SSIM':<15} {'时间(秒)':<15} {'迭代次数':<15} {'每轮时间(ms)'}")
    print("-"*90)
    
    for algo_name in ['ISTA-TV', 'FISTA-TV', 'ADMM-TV', 'BM3D']:
        psnr = metrics.peak_signal_noise_ratio(img, results[algo_name])
        ssim = ssim_values[algo_name]
        time_used = times[algo_name]
        
        if algo_name != 'BM3D':
            n_iter = times[f"{algo_name.split('-')[0]}-iter"]
            time_per_iter = (time_used / n_iter) * 1000
            print(f"{algo_name:<15} {psnr:<15.2f} {ssim:<15.4f} {time_used:<15.2f} {n_iter:<15} {time_per_iter:<15.2f}")
        else:
            print(f"{algo_name:<15} {psnr:<15.2f} {ssim:<15.4f} {time_used:<15.2f} {'N/A':<15} {'N/A':<15}")


if __name__ == "__main__":
    main()
