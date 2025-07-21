import random
import math
import torch
import numpy as np

def get_f2fd_pair(vol, bernoulli_mask_ratio=0.5, phase_inversion_ratio=0.1, min_mask_radius=0.05, max_mask_radius=0.1, patch_size=8):
    """Applies Fourier-based perturbations to create paired noisy patches."""
    
    input_is_ndarray = False
    if isinstance(vol, np.ndarray):
        vol = torch.from_numpy(vol)
        input_is_ndarray = True
    if not isinstance(vol, torch.Tensor):
        raise TypeError("Input volume must be a numpy array or a torch tensor.")            
    
    fft_patch = torch.fft.rfftn(vol)
    fft_patch = torch.fft.fftshift(fft_patch, dim=(-3, -2))
    
    _mask = lambda x: mask(
        x, 
        bernoulli_mask_ratio=bernoulli_mask_ratio,
        phase_inversion_ratio=phase_inversion_ratio,
        min_mask_radius=min_mask_radius,
        max_mask_radius=max_mask_radius,
        patch_size=patch_size,
    )
    
    fft_patch1 = _mask(fft_patch)
    fft_patch2 = _mask(fft_patch)

    # Inverse FFT
    ifft_patch1 = torch.fft.irfftn(torch.fft.ifftshift(fft_patch1, dim=(-3, -2)), s=vol.shape).float()
    ifft_patch2 = torch.fft.irfftn(torch.fft.ifftshift(fft_patch2, dim=(-3, -2)), s=vol.shape).float()
    
    if input_is_ndarray:
        ifft_patch1 = ifft_patch1.numpy()
        ifft_patch2 = ifft_patch2.numpy()
    
    return ifft_patch1, ifft_patch2

def mask(vol_fft, bernoulli_mask_ratio, phase_inversion_ratio, min_mask_radius, max_mask_radius, patch_size=8):
    # Patch-based Bernoulli Masking (8x8x8 patches)
    size = vol_fft.shape[0]

    mask_shape = (size // patch_size + 1, size // patch_size + 1, 1 + (size // 2 + 1) // patch_size + 1)
    patch_mask = (torch.rand(*mask_shape, device=vol_fft.device) > bernoulli_mask_ratio).float()
    
    # Expand mask to match fft_patch shape
    # bernoulli_mask = patch_mask.numpy().repeat(patch_size, axis=0)
    # bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=1)
    # bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=2)
    bernoulli_mask = patch_mask.repeat((patch_size, 1, 1))
    bernoulli_mask = bernoulli_mask.repeat(1, patch_size, 1)
    bernoulli_mask = bernoulli_mask.repeat(1, 1, patch_size)
    bernoulli_mask = bernoulli_mask[:size, :size, :size // 2 + 1]  # Ensure matching shape
    
    # fft_masked = fft_patch * bernoulli_mask
    
    # Random Phase Inversion
    phase_flip = (torch.rand(*vol_fft.real.shape, device=vol_fft.device) < phase_inversion_ratio).float()
    fft_patch_inverted = vol_fft * ((-1) ** phase_flip)
    
    # Spherical Mask to keep low frequencies
    center = size // 2
    # radius = random.randint(int(0.05 * size), int(0.1 * size))
    radius = random.randint(int(min_mask_radius * size), int(max_mask_radius * size))
    x, y, z = torch.meshgrid(
        torch.arange(size, device=vol_fft.device), 
        torch.arange(size, device=vol_fft.device), 
        torch.arange(size // 2 + 1, device=vol_fft.device), 
        indexing='ij'
    )
    mask = ((x - center) ** 2 + (y - center) ** 2 + (z) ** 2) < (radius ** 2)
    
    mask = mask.float()

    overall_mask = torch.logical_or(mask, bernoulli_mask)
    # fft_masked = torch.where(mask, fft_masked, torch.zeros_like(fft_masked))
    return fft_patch_inverted * overall_mask.float()



# %