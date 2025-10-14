"""
Peak finding Ray task for CPU-based post-processing.

This module provides a stateless Ray task that performs peak finding on
segmentation maps using scipy.ndimage.
"""

import ray
import numpy as np
from scipy import ndimage
import torch


@ray.remote
def process_samples_task(samples_ref, structure_ref):
    """
    Stateless Ray task for peak finding using scipy.ndimage.

    This task processes a mini-batch of samples, converting logits to segmentation
    maps and finding peaks using connected component labeling and center of mass
    calculation.

    Args:
        samples_ref: ObjectRef to mini-batch of sample logits with shape (N, num_classes, H, W)
        structure_ref: ObjectRef to shared 8-connectivity structure for ndimage.label

    Returns:
        List of peak positions for each sample: [[[p, y, x], ...], ...]
        where p is the sample/panel index, y and x are pixel coordinates.

    Note:
        - Uses ObjectRefs for zero-copy data sharing
        - Performs softmax + argmax to convert logits to binary segmentation
        - Uses scipy.ndimage.label for connected component labeling
        - Uses scipy.ndimage.center_of_mass for peak localization
    """
    # Dereference from Ray object store
    samples = ray.get(samples_ref)  # Mini-batch (e.g., 2-4 samples)
    structure = ray.get(structure_ref)  # Shared structure

    all_peaks = []

    for sample_idx, sample_logits in enumerate(samples):
        # Stage 1: Logits → Segmentation Map
        # Input: (num_classes=2, H, W) - class 0=background, class 1=peak
        # Output: (H, W) binary mask
        if isinstance(sample_logits, torch.Tensor):
            seg_map = sample_logits.softmax(dim=0).argmax(dim=0).cpu().numpy()
        else:
            # Already numpy array
            seg_map = sample_logits.argmax(axis=0)

        # Stage 2: Connected Component Labeling
        # Find distinct peak regions using 8-connectivity
        labeled_map, num_peaks = ndimage.label(seg_map, structure)

        # Stage 3: Center of Mass for each peak
        if num_peaks > 0:
            peak_coords = ndimage.center_of_mass(
                seg_map.astype(np.float32),
                labeled_map.astype(np.float32),
                np.arange(1, num_peaks + 1)
            )

            # Convert to [p, y, x] format (p = panel/sample index)
            # Filter out invalid coordinates
            peaks = []
            for coords in peak_coords:
                if isinstance(coords, tuple) and len(coords) == 2:
                    y, x = coords
                    peaks.append([sample_idx, float(y), float(x)])
                elif isinstance(coords, (list, np.ndarray)) and len(coords) == 2:
                    y, x = coords
                    peaks.append([sample_idx, float(y), float(x)])
        else:
            peaks = []

        all_peaks.append(peaks)

    return all_peaks
