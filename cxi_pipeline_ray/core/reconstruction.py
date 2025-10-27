"""
Detector image reconstruction and metadata conversion utilities.

This module provides utilities to reconstruct detector images from preprocessed
format back to original size, and convert physics metadata to CXI-compatible formats.
"""

import numpy as np
import ray
from typing import Union


def reconstruct_detector_image(output) -> np.ndarray:
    """
    Reconstruct detector image from preprocessed format to original size.

    This function reverses the preprocessing pipeline:
    1. Reshape: (B*C, 1, H, W) → (B, C, H, W)
    2. Unpad: (B, C, H, W) → (B, C, H_orig, W_orig)

    Args:
        output: PipelineOutput from Q2 with original_image_ref and preprocessing_metadata

    Returns:
        detector_image: (B, C, H_orig, W_orig) numpy array in original coordinates

    Raises:
        ValueError: If preprocessing_metadata is missing or invalid

    Example:
        >>> output = q2_manager.get()
        >>> detector_image = reconstruct_detector_image(output)
        >>> assert detector_image.shape == (8, 16, 352, 384)  # B=8, C=16 panels
    """
    # Check if preprocessing metadata is available
    if output.preprocessing_metadata is None:
        # No preprocessing was applied - return image as-is
        if output.original_image_ref is None:
            raise ValueError("Both preprocessing_metadata and original_image_ref are None")
        return ray.get(output.original_image_ref)

    # Get preprocessed detector image
    if output.original_image_ref is None:
        raise ValueError("original_image_ref is None but preprocessing_metadata is present")

    detector_image = ray.get(output.original_image_ref)  # (B*C, 1, H, W)

    # Extract shape information from metadata
    B, C, H_orig, W_orig = output.preprocessing_metadata.original_shape
    BC, _, H, W = output.preprocessing_metadata.preprocessed_shape

    # Validate shapes
    if detector_image.shape != (BC, 1, H, W):
        raise ValueError(
            f"Image shape {detector_image.shape} doesn't match "
            f"preprocessed_shape {(BC, 1, H, W)} from metadata"
        )

    # Step 1: Reshape (B*C, 1, H, W) → (B, C, H, W)
    detector_image_reshaped = detector_image.reshape(B, C, H, W)

    # Step 2: Unpad (B, C, H, W) → (B, C, H_orig, W_orig)
    # Bottom-right padding means original data is at [0:H_orig, 0:W_orig]
    detector_image_original = detector_image_reshaped[:, :, :H_orig, :W_orig]

    return detector_image_original


def wavelength_to_energy(wavelength: Union[float, np.ndarray], unit: str = 'angstrom') -> Union[float, np.ndarray]:
    """
    Convert photon wavelength to photon energy.

    Uses the relationship: E = hc/λ
    Where:
    - h = Planck's constant = 4.135667696e-15 eV·s
    - c = speed of light = 299792458 m/s
    - hc = 1.23984193 eV·μm = 12398.4193 eV·Å

    Args:
        wavelength: Wavelength value(s) to convert
        unit: Unit of wavelength. Options:
              - 'angstrom' or 'Å': Angstroms (default, common in X-ray crystallography)
              - 'nm': Nanometers
              - 'um' or 'μm': Micrometers

    Returns:
        Photon energy in eV (electron volts)

    Examples:
        >>> wavelength_to_energy(1.5)  # 1.5 Å X-ray
        8265.6128666...

        >>> wavelength_to_energy(np.array([1.0, 1.5, 2.0]))  # Multiple wavelengths
        array([12398.4193,  8265.6129,  6199.2097])

        >>> wavelength_to_energy(500, unit='nm')  # 500 nm visible light
        2.47968386
    """
    # Conversion constants (hc in different units)
    HC_EV_ANGSTROM = 12398.4193  # eV·Å
    HC_EV_NM = 1239.84193        # eV·nm
    HC_EV_UM = 1.23984193        # eV·μm

    # Select appropriate constant based on unit
    if unit in ('angstrom', 'Å', 'A'):
        hc = HC_EV_ANGSTROM
    elif unit == 'nm':
        hc = HC_EV_NM
    elif unit in ('um', 'μm', 'micron'):
        hc = HC_EV_UM
    else:
        raise ValueError(f"Unknown wavelength unit: {unit}. Use 'angstrom', 'nm', or 'um'")

    # Handle zero wavelength (avoid division by zero)
    if isinstance(wavelength, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            energy = hc / wavelength
            energy = np.where(wavelength == 0, 0.0, energy)
    else:
        if wavelength == 0:
            return 0.0
        energy = hc / wavelength

    return energy


def energy_to_wavelength(energy: Union[float, np.ndarray], unit: str = 'angstrom') -> Union[float, np.ndarray]:
    """
    Convert photon energy to wavelength.

    Inverse of wavelength_to_energy. Uses: λ = hc/E

    Args:
        energy: Photon energy in eV (electron volts)
        unit: Desired wavelength unit ('angstrom', 'nm', or 'um')

    Returns:
        Wavelength in specified unit

    Example:
        >>> energy_to_wavelength(8265.6)  # 8.27 keV X-ray
        1.49999...  # ~1.5 Å
    """
    # Use same conversion as wavelength_to_energy (E and λ are symmetric)
    return wavelength_to_energy(energy, unit=unit)


def get_num_panels_from_metadata(output) -> int:
    """
    Extract number of panels (C dimension) from preprocessing metadata.

    This is used for unpacking (B*C) peaks back into (B, C) event/panel structure.

    Args:
        output: PipelineOutput with preprocessing_metadata

    Returns:
        Number of panels per event (C dimension)

    Raises:
        ValueError: If metadata is missing

    Example:
        >>> output = q2_manager.get()
        >>> num_panels = get_num_panels_from_metadata(output)
        >>> print(num_panels)  # 16 for ePix10k2M
        16
    """
    if output.preprocessing_metadata is None:
        raise ValueError("preprocessing_metadata is None - cannot determine number of panels")

    B, C, H_orig, W_orig = output.preprocessing_metadata.original_shape
    return C
