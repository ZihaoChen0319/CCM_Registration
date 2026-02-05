# Surface Distance related are based on https://github.com/google-deepmind/surface-distance

import numpy as np
import scipy
import torch

from evaluator.surface_distance import compute_average_surface_distance, compute_robust_hausdorff, \
    compute_surface_overlap_at_tolerance, compute_surface_dice_at_tolerance, compute_dice_coefficient, \
    compute_surface_distances


def jacobian_determinant(flow):
    _, _, H, W, D = flow.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape((1, 3, 1, 1))
    grady = np.array([-0.5, 0, 0.5]).reshape((1, 1, 3, 1))
    gradz = np.array([-0.5, 0, 0.5]).reshape((1, 1, 1, 3))

    gradx_disp = np.stack([scipy.ndimage.correlate(flow[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(flow[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(flow[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(flow[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(flow[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(flow[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(flow[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(flow[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(flow[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, ...] * (jacobian[1, 1, ...] * jacobian[2, 2, ...] - jacobian[1, 2, ...] * jacobian[2, 1, ...]) - \
             jacobian[1, 0, ...] * (jacobian[0, 1, ...] * jacobian[2, 2, ...] - jacobian[0, 2, ...] * jacobian[2, 1, ...]) + \
             jacobian[2, 0, ...] * (jacobian[0, 1, ...] * jacobian[1, 2, ...] - jacobian[0, 2, ...] * jacobian[1, 1, ...])

    return jacdet


def compute_SDlogJ(flow):
    jac_det = (jacobian_determinant(flow) + 3).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det)
    single_value = log_jac_det.std()
    return single_value


def compute_dice(fixed, moving_warped, labels=None):
    if labels is None:
        labels = np.arange(np.max(fixed) + 1).astype(int)
    dice = []
    for i in labels:
        if (fixed[:, i].sum() == 0) or (moving_warped[:, i].sum() == 0):
            dice.append(np.nan)
        else:
            dice.append(compute_dice_coefficient(fixed[:, i].astype(bool), moving_warped[:, i].astype(bool)))
    mean_dice = np.nanmean(dice)
    return mean_dice, dice


def compute_dice_torch(fixed, moving_warped):
    ndims = len(list(fixed.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (fixed * moving_warped).sum(dim=vol_axes)
    bottom = torch.clamp((fixed + moving_warped).sum(dim=vol_axes), min=1e-5)
    dice = top / bottom
    mean_dice = dice.mean(dim=1)
    return mean_dice, dice


def compute_hd95(fixed, moving_warped, labels=None):
    if labels is None:
        labels = np.arange(np.max(fixed) + 1).astype(int)
    hd95 = []
    for i in labels:
        if (fixed[:, i].sum() == 0) or (moving_warped[:, i].sum() == 0):
            hd95.append(np.nan)
        else:
            if len(fixed.shape) == 4:
                spacing_mm_current = np.ones(2)
            elif len(fixed.shape) == 5:
                spacing_mm_current = np.ones(3)
            else:
                spacing_mm_current = None
            surface_distances_current = compute_surface_distances(
                fixed[:, i].astype(bool).squeeze(), moving_warped[:, i].astype(bool).squeeze(),
                spacing_mm=spacing_mm_current)
            hd95.append(compute_robust_hausdorff(surface_distances_current, 95.))
    mean_hd95 = np.nanmean(hd95)
    return mean_hd95, hd95



