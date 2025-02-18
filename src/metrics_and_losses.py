import numpy as np
from typing import Sequence, Dict, Optional, List, Any, Tuple, Callable
import sklearn.metrics


import torch
import torch.nn.functional as F


def mse_skipnan(pred: torch.Tensor, target: torch.Tensor, *args) -> torch.Tensor:
    mask = torch.isnan(target)
    pred_flat = torch.masked_select(pred, ~mask)
    target_flat = torch.masked_select(target, ~mask)
    raw = F.mse_loss(pred_flat, target_flat, reduction='none')
    loss = torch.sum(raw) / torch.numel(target)
    return loss


def mse_skipnan_with_stddev(pred: torch.Tensor, target: torch.Tensor,
                            target_stddev: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape, f"{pred.shape=} {target.shape=}"
    mask_target = ~torch.isnan(target)
    mask_stddev = ~torch.isnan(target_stddev)
    mask = torch.logical_and(mask_target, mask_stddev)
    pred_flat = torch.masked_select(pred, mask)
    target_flat = torch.masked_select(target, mask)
    target_stddev_flat = torch.masked_select(target_stddev, mask)
    raw_loss = F.mse_loss(pred_flat, target_flat, reduction='none')
    # Use target_stddev_flat
    alpha = 0.1
    coef = alpha / (alpha + target_stddev_flat)
    scaled_loss = coef * raw_loss
    loss = torch.sum(scaled_loss) / torch.numel(target)
    return loss


def bce_skipnan(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape, (
        f"Shapes do not match: pred.shape={pred.shape}, target.shape={target.shape}")
    mask = ~torch.isnan(target)
    pred_masked = torch.masked_select(pred, mask)
    target_masked = torch.masked_select(target, mask)
    bce_per_sample = F.binary_cross_entropy(pred_masked, target_masked, reduction='none')
    loss = torch.sum(bce_per_sample) / torch.numel(target)
    return loss

def raw_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raw = F.mse_loss(pred, target, reduction='none')
    return raw


def regular_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return np.mean((pred - target)**2).item()


def inlier_indicies(pred: np.ndarray, target: np.ndarray,
                     inlier_frac: float = 0.9) -> np.ndarray:
    num_inliers = max(1, int(inlier_frac * len(target)))
    inliers = np.argsort(np.abs(pred - target))[:num_inliers]
    return inliers


def stable_mse(pred_flat: np.ndarray, target_flat: np.ndarray) -> float:
    inliers = inlier_indicies(pred_flat, target_flat)
    stable_mse = np.mean((pred_flat[inliers] - target_flat[inliers])**2).item()
    return stable_mse


def group_by_properties(block_pred: np.ndarray, block_target: np.ndarray) \
        -> List[Tuple[np.ndarray, np.ndarray]]:

    lst = []
    for i_prop in range(block_target.shape[1]):
        pred = block_pred[:, i_prop]
        target = block_target[:, i_prop]
        mask = np.isnan(target)
        pred_flat = pred[~mask]
        target_flat = target[~mask]
        lst.append((pred_flat, target_flat))
    return lst


def per_prop_mse(block_pred: np.ndarray, block_target: np.ndarray) -> List[float]:
    mse_list = []
    for i_prop, (pred, target) in enumerate(
            group_by_properties(block_pred, block_target)):

        mask = np.isnan(target)
        pred_flat = pred[~mask]
        target_flat = target[~mask]

        reg_mse = regular_mse(pred_flat, target_flat)
        mse_list.append(reg_mse)
    return mse_list


def per_prop_stable_mse(block_pred: np.ndarray, block_target: np.ndarray) -> List[float]:
    stable_mse_list = []
    for i_prop, (pred, target) in enumerate(
            group_by_properties(block_pred, block_target)):

        mask = np.isnan(target)
        pred_flat = pred[~mask]
        target_flat = target[~mask]

        st_mse = stable_mse(pred_flat, target_flat)
        stable_mse_list.append(st_mse)

    return stable_mse_list


def per_prop_r2(block_pred: np.ndarray, block_target: np.ndarray,
                is_stable: bool = False) -> List[float]:
    r2_list = []
    for i_prop, (pred, target) in enumerate(
            group_by_properties(block_pred, block_target)):

        mask = np.isnan(target)
        pred_flat = pred[~mask]
        target_flat = target[~mask]
        if len(pred_flat) <= 2:
            r2 = np.nan
        else:
            if is_stable:
                inliers = inlier_indicies(pred_flat, target_flat)
                r2 = sklearn.metrics.r2_score(pred[inliers], target[inliers])
            else:
                r2 = sklearn.metrics.r2_score(target_flat, pred_flat)
        r2_list.append(r2)
    return r2_list


def per_prop_roc_auc(block_pred: np.ndarray, block_target: np.ndarray) -> List[float]:
    roc_auc_list = []
    for i_prop, (pred, target) in enumerate(
            group_by_properties(block_pred, block_target)):
        mask = np.isnan(target)
        pred_flat = pred[~mask]
        target_flat = target[~mask]
        unique_classes = np.unique(target_flat)
        if len(unique_classes) < 2:
            roc_auc = 0.0  
        else:
            roc_auc = sklearn.metrics.roc_auc_score(target_flat, pred_flat)
        roc_auc_list.append(roc_auc)
    return roc_auc_list