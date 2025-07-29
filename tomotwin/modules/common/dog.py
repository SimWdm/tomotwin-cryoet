import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter


def non_maximum_suppression_3d(x, d, scale=1.0, threshold=-np.inf):
    ## enumerate coordinate deltas within d
    r = scale*d/2
    width = int(np.ceil(r))
    A = np.arange(-width,width+1)
    ii,jj,kk = np.meshgrid(A, A, A)
    mask = (ii**2 + jj**2 + kk**2) <= r*r
    ii = ii[mask]
    jj = jj[mask]
    kk = kk[mask]
    zstride = x.shape[1]*x.shape[2]
    ystride = x.shape[2]
    coord_deltas = ii*zstride + jj*ystride + kk
    
    A = x.ravel()
    I = np.argsort(A, axis=None)[::-1] # reverse to sort in descending order
    S = set() # the set of suppressed coordinates

    scores = np.zeros(len(A), dtype=np.float32)
    coords = np.zeros((len(A),3), dtype=np.int32)

    j = 0
    for i in I:
        if A[i] <= threshold:
            break
        if i not in S:
            ## coordinate i is next center
            zz,yy,xx = np.unravel_index(i, x.shape)
            scores[j] = A[i]
            coords[j,0] = xx
            coords[j,1] = yy
            coords[j,2] = zz
            j += 1
            ## add coordinates within d of i to the suppressed set
            for delta in coord_deltas:
                S.add(i + delta)
    
    return scores[:j], coords[:j]

def _nms_xy(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool3d(
        heat, (1, kernel, kernel), stride=1, padding=(0,pad,pad))
    keep = (hmax == heat).float()
    return heat * keep


def get_potential_coords_pyramid(rec, sigmas = [2,4], num_pyramid = 3, kernel = 3, bounds = [10,30,30]):
    # if rec is np array, convert to torch tensor
    if isinstance(rec, np.ndarray):
        rec = torch.from_numpy(rec).float()
    elif not isinstance(rec, torch.Tensor):
        raise TypeError("Input must be a numpy array or a torch tensor.")
    
    ims = []
    z, r, c = rec.shape
    bound_z, bound_x, bound_y = bounds
    # if r > 512 and c > 512:
    #     bound_x, bound_y = bound_x * 2, bound_y * 2
    #     # do 2x downsampling first 
    #     down_rec = []
    #     for i in range(z):
    #         down_rec.append(rescale(rec[i],0.5,anti_aliasing=True))
    #     down_rec = np.asarray(down_rec)
    # rec = down_rec
    # sigma = sigma_init
    num_pyramid = len(sigmas)
    for i in range(num_pyramid):
        # sigma = sigma*(i+1)
        sigma = sigmas[i]
        im = gaussian_filter(rec, sigma)
        ims.append(im)
        # sigma *= 2
    diff_all = []
    for i in range(num_pyramid-1):
        diff = ims[i+1] - ims[i]
        diff[:bound_z,:,:] = 0
        diff[-bound_z:,:,:]= 0
        diff[:,:bound_x,:] = 0 
        diff[:,-bound_x:,:] = 0
        diff[:,:,:bound_y] = 0
        diff[:,:,-bound_y:] = 0
        diff = torch.as_tensor(diff)
        diff = diff.unsqueeze(0).unsqueeze(0) 
        nms_diff_xy = _nms_xy(diff, kernel=kernel) 
        nms_diff_xy = nms_diff_xy.squeeze().numpy()
       
        diff_all.append(nms_diff_xy)
    diff_alls = np.stack(diff_all, axis=0) 
    nms_diff_xy = np.max(diff_alls, axis=0)
    nms_diff_xy = torch.as_tensor(nms_diff_xy)
     
    mean_nms = nms_diff_xy[torch.where(nms_diff_xy > 0)].mean().item()
    std_nms_half = nms_diff_xy[torch.where(nms_diff_xy > 0)].std().item() 
    cutoff_score = mean_nms + std_nms_half*0.5
    nms_diff_xy = nms_diff_xy.squeeze().numpy()
    scores, coords = non_maximum_suppression_3d(nms_diff_xy, 14, threshold=cutoff_score)

    return scores, coords

def get_dog_proposals(tomo, sigmas=[4, 8], downsampling = 2):   
    # if tomo is np array, convert to torch tensor
    if isinstance(tomo, np.ndarray):
        tomo = torch.from_numpy(tomo).float()
    elif not isinstance(tomo, torch.Tensor):
        raise TypeError("Input must be a numpy array or a torch tensor.")
    
    tomo = torch.nn.functional.avg_pool3d(tomo.unsqueeze(0), kernel_size=downsampling, stride=downsampling, padding=0).squeeze(0)
    scores, coords = get_potential_coords_pyramid(tomo, sigmas=sigmas, num_pyramid=2, kernel=3, bounds=[17, 17, 17])
    coords *= downsampling
    coords = coords.astype(int)
    coords = pd.DataFrame(coords, columns=["X", "Y", "Z"])
    return coords


