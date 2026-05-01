import numpy as np
import bm3d
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


def pnp_admm_mri(kspace_und, mask, rho0=0.01, lam=0.0005, gamma=1.05, tol=1e-3, max_iter=100):
    """
    Plug-and-Play ADMM for MRI reconstruction.
    """
    x = np.fft.ifft2(kspace_und)
    x = np.abs(x)
    x = x / (np.max(x) + 1e-10)
    
    v = x.copy()
    u = np.zeros_like(x, dtype=np.complex128)
    rho = rho0

    for k in range(max_iter):
        x_prev = x.copy()
        v_prev = v.copy()

        # Step 1: x-update in k-space
        Fv_u = np.fft.fft2(v - u)
        kspace_x = (mask * kspace_und + rho * Fv_u) / (mask + rho + 1e-10)
        x = np.abs(np.fft.ifft2(kspace_x))

        # Step 2: v-update (BM3D denoising)
        sigma_k = np.sqrt(lam / rho)
        v_tilde = x + np.abs(u)
        v_tilde_norm = (v_tilde - v_tilde.min()) / (v_tilde.max() - v_tilde.min() + 1e-10)
        v_tilde_uint8 = np.clip(v_tilde_norm * 255, 0, 255).astype(np.uint8)
        # sigma_k is small, scale appropriately for BM3D
        v_uint8 = bm3d.bm3d(v_tilde_uint8, sigma_k * 255)
        v = v_uint8.astype(np.float64) / 255.0
        v = v * (v_tilde.max() - v_tilde.min()) + v_tilde.min()

        # Step 3: u-update
        u = u + (x - v)

        # Step 4: rho update
        rho = rho * gamma

        delta = (np.linalg.norm(x - x_prev) + np.linalg.norm(v - v_prev)) / np.sqrt(x.size)
        if delta < tol:
            print(f'  Converged at iter {k+1}')
            break

    return np.abs(x)


print('Loading MRI data...')
img = nib.load('../week11/data/stanford_hardi/dwi.nii').get_fdata()
clean_slice = img[:, :, 40, 50].astype(np.float64)
clean_slice = clean_slice / np.max(clean_slice)

kspace_full = np.fft.fft2(clean_slice)

np.random.seed(42)
h, w = clean_slice.shape
mask = np.zeros((h, w))
lines = np.sort(np.random.choice(w, int(w * 0.3), replace=False))
mask[:, lines] = 1

kspace_und = kspace_full * mask

zf_recon = np.abs(np.fft.ifft2(kspace_und))
zf_recon = zf_recon / np.max(zf_recon)
print(f'Zero-filled PSNR: {psnr(clean_slice, zf_recon):.2f} dB')

params = [(0.01, 0.0005), (0.005, 0.0002), (0.02, 0.001)]
results = {}
for rho0, lam in params:
    print(f'PnP-ADMM (rho0={rho0}, lam={lam}):')
    recon = pnp_admm_mri(kspace_und, mask, rho0=rho0, lam=lam)
    recon = recon / np.max(recon)
    results[(rho0, lam)] = recon
    print(f'  PSNR: {psnr(clean_slice, recon):.2f} dB')

best = max(results, key=lambda k: psnr(clean_slice, results[k]))
best_recon = results[best]
print(f'\nBest: rho0={best[0]}, lam={best[1]}, PSNR={psnr(clean_slice, best_recon):.2f} dB')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(clean_slice, cmap='gray')
axes[0].set_title('Ground Truth')
axes[0].axis('off')
axes[1].imshow(zf_recon, cmap='gray')
axes[1].set_title(f'Zero-filled ({psnr(clean_slice, zf_recon):.1f} dB)')
axes[1].axis('off')
axes[2].imshow(best_recon, cmap='gray')
axes[2].set_title(f'PnP-ADMM ({psnr(clean_slice, best_recon):.1f} dB)')
axes[2].axis('off')
axes[3].imshow(np.abs(clean_slice - best_recon), cmap='hot')
axes[3].set_title('Error Map')
axes[3].axis('off')
plt.tight_layout()
plt.savefig('pnp_mri_result.png', dpi=150)
plt.show()
print('Done!')