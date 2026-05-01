import numpy as np
import bm3d
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def pnp_admm_denoise(noisy, rho0=0.1, lam=0.05, gamma=1.01, tol=1e-3, max_iter=50):
    """
    Plug-and-Play ADMM for image denoising.
    f(x) = (1/2)||x - noisy||^2   (data fidelity)
    g(v) = denoiser prior (BM3D)
    """
    x = noisy.copy()
    v = noisy.copy()
    u = np.zeros_like(noisy)
    rho = rho0

    for k in range(max_iter):
        x_prev = x.copy()
        v_prev = v.copy()
        u_prev = u.copy()

        # Step 1: x-update (data fidelity)
        # x = (noisy + rho*(v - u)) / (1 + rho)
        x = (noisy + rho * (v - u)) / (1.0 + rho)

        # Step 2: v-update (denoising with BM3D)
        sigma_k = np.sqrt(lam / rho)
        v_tilde = x + u
        v_tilde_clipped = np.clip(v_tilde, 0, 1)
        v = bm3d.bm3d((v_tilde_clipped * 255).astype(np.uint8), sigma_k * 255) / 255.0

        # Step 3: u-update
        u = u + (x - v)

        # Step 4: rho update (continuation)
        rho = gamma * rho

        # Convergence check
        delta = (np.linalg.norm(x - x_prev) + 
                 np.linalg.norm(v - v_prev) + 
                 np.linalg.norm(u - u_prev)) / np.sqrt(x.size)
        if delta < tol:
            print(f'Converged at iteration {k+1}')
            break

    return v


# Load test image
from skimage.data import camera
clean = img_as_float(camera())

# Add Gaussian noise
sigma = 25 / 255
np.random.seed(42)
noisy = clean + sigma * np.random.randn(*clean.shape)
noisy = np.clip(noisy, 0, 1)

print(f'Noisy PSNR: {psnr(clean, noisy):.2f} dB')
print(f'Noisy SSIM: {ssim(clean, noisy, data_range=1):.4f}')

# Run PnP-ADMM with different parameters
params = [
    (0.1, 0.05),
    (0.2, 0.03),
    (0.5, 0.01),
]

for rho0, lam in params:
    print(f'\nRunning PnP-ADMM (rho0={rho0}, lambda={lam})...')
    denoised = pnp_admm_denoise(noisy, rho0=rho0, lam=lam)
    denoised = np.clip(denoised, 0, 1)
    print(f'Denoised PSNR: {psnr(clean, denoised):.2f} dB')
    print(f'Denoised SSIM: {ssim(clean, denoised, data_range=1):.4f}')

# Use best params for final visualization
print('\n--- Final result with best params ---')
best_denoised = pnp_admm_denoise(noisy, rho0=0.1, lam=0.05)
best_denoised = np.clip(best_denoised, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(clean, cmap='gray')
axes[0].set_title('Ground Truth')
axes[0].axis('off')
axes[1].imshow(noisy, cmap='gray')
axes[1].set_title(f'Noisy ({sigma*255:.0f}/255, {psnr(clean, noisy):.1f}dB)')
axes[1].axis('off')
axes[2].imshow(best_denoised, cmap='gray')
axes[2].set_title(f'PnP-ADMM ({psnr(clean, best_denoised):.1f}dB)')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('pnp_denoise_result.png', dpi=150)
plt.show()
print('Result saved to pnp_denoise_result.png')