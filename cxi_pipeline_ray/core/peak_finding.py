"""
Peak finding using scipy.ndimage for CXI post-processing.

This module provides a plain function (no Ray, no torch) that performs peak finding
on segmentation logits using connected component labeling and center of mass.
"""

import numpy as np
from scipy import ndimage


def find_peaks_numpy(logits_2d: np.ndarray, return_seg_map: bool = False):
    """
    Find peaks from 2-class logits using scipy.ndimage.label.

    Args:
        logits_2d: (2, H, W) logits from model
        return_seg_map: If True, return (peaks, seg_map) tuple

    Returns:
        peaks: (N, 3) array of [panel_idx=0, y, x]
        OR
        (peaks, seg_map): If return_seg_map=True, where seg_map is (H, W) uint8 with values 0 or 1
    """
    # Binary segmentation via argmax (class 0=background, class 1=peak)
    seg_map = np.argmax(logits_2d, axis=0)  # (H, W) with values 0 or 1

    peak_mask = (seg_map == 1)

    # Find connected components (8-connectivity)
    structure = np.ones((3, 3), dtype=np.float32)
    labeled, num_features = ndimage.label(peak_mask, structure=structure)

    if num_features == 0:
        peaks = np.array([]).reshape(0, 3)
    else:
        # Find center of mass for each component
        peaks = []
        for label_id in range(1, num_features + 1):
            component_mask = labeled == label_id
            y_coords, x_coords = np.where(component_mask)

            # Center of mass
            y_center = y_coords.mean()
            x_center = x_coords.mean()

            peaks.append([0, y_center, x_center])  # panel_idx=0 for single panel

        peaks = np.array(peaks, dtype=np.float32)

    if return_seg_map:
        return peaks, seg_map.astype(np.uint8)
    else:
        return peaks
