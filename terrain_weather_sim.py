#!/usr/bin/env python3
"""
Quantum-Seeded Terrain and Weather Simulation

This script simulates quantum-seeded terrain generation, including detailed 
stratigraphy, geological feature classification (wind barriers, channels, basins),
orographic low-pressure computation, and multi-year climate/weather patterns 
with rainfall accumulation.

Components:
- Quantum RNG using Qiskit (with NumPy fallback)
- Terrain generation (fractional surface, domain warp, ridged mix)
- Stratigraphy generation with geological layers
- Wind structure classification
- Orographic low-pressure computation
- Climate/weather trend module
- Storm scheduling and weather field generation
- Rainfall accumulation
- Visualization
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

# Attempt to import Qiskit for quantum RNG
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available. Using NumPy fallback for RNG.")

# Attempt to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.patches import Patch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Plotting disabled.")


# =============================================================================
# QUANTUM RNG MODULE
# =============================================================================

def qrng_uint32(n_values: int = 1, shots_per_circuit: int = 1024) -> NDArray[np.uint32]:
    """
    Generate quantum random 32-bit unsigned integers using Qiskit.
    
    Uses a quantum circuit with 32 qubits, each in superposition, to generate
    truly random bits.
    """
    if not QISKIT_AVAILABLE:
        return np.random.default_rng().integers(0, 2**32, size=n_values, dtype=np.uint32)
    
    results = []
    remaining = n_values
    
    while remaining > 0:
        # Create a 32-qubit circuit
        qc = QuantumCircuit(32, 32)
        qc.h(range(32))  # Apply Hadamard to all qubits
        qc.measure(range(32), range(32))
        
        # Execute
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(qc, shots=min(shots_per_circuit, remaining))
        counts = job.result().get_counts()
        
        # Convert bitstrings to integers
        for bitstring, count in counts.items():
            val = int(bitstring, 2)
            for _ in range(min(count, remaining)):
                results.append(val)
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
    
    return np.array(results[:n_values], dtype=np.uint32)


def rng_from_qrng(seed_count: int = 4) -> np.random.Generator:
    """Create a NumPy Generator seeded with quantum random numbers."""
    seeds = qrng_uint32(seed_count)
    # Create a seed sequence from quantum random values
    seed_val = int(seeds[0]) if len(seeds) > 0 else 42
    return np.random.default_rng(seed_val)


def quantum_bits(n: int) -> NDArray[np.uint8]:
    """Generate n quantum random bits."""
    if not QISKIT_AVAILABLE:
        return np.random.default_rng().integers(0, 2, size=n, dtype=np.uint8)
    
    n_words = (n + 31) // 32
    words = qrng_uint32(n_words)
    bits = np.unpackbits(words.view(np.uint8))[:n]
    return bits.astype(np.uint8)


def quantum_uniforms(n: int, rng: Optional[np.random.Generator] = None) -> NDArray[np.float64]:
    """Generate n quantum-seeded uniform random numbers in [0, 1)."""
    if rng is None:
        rng = rng_from_qrng()
    return rng.random(n)


def quantum_integers(low: int, high: int, n: int, rng: Optional[np.random.Generator] = None) -> NDArray[np.int64]:
    """Generate n quantum-seeded random integers in [low, high)."""
    if rng is None:
        rng = rng_from_qrng()
    return rng.integers(low, high, size=n)


class _QuantumRNGWrapper:
    """Wrapper class for quantum_uniforms to be used in simulation driver."""
    
    def __init__(self, base_rng: Optional[np.random.Generator] = None):
        self.base_rng = base_rng if base_rng is not None else rng_from_qrng()
    
    def __call__(self, n: int) -> NDArray[np.float64]:
        return quantum_uniforms(n, self.base_rng)
    
    def uniform(self, low: float = 0.0, high: float = 1.0, size: int = 1) -> NDArray[np.float64]:
        """Generate uniform random numbers in [low, high)."""
        u = self(size)
        return low + (high - low) * u
    
    def integers(self, low: int, high: int, size: int = 1) -> NDArray[np.int64]:
        """Generate random integers in [low, high)."""
        return quantum_integers(low, high, size, self.base_rng)


# =============================================================================
# TERRAIN GENERATION MODULE
# =============================================================================

def _fft2_freq_grid(nx: int, ny: int) -> Tuple[NDArray, NDArray]:
    """Generate frequency grids for FFT operations."""
    fx = np.fft.fftfreq(nx)
    fy = np.fft.fftfreq(ny)
    return np.meshgrid(fx, fy, indexing='ij')


def fractional_surface(
    nx: int,
    ny: int,
    beta: float = 2.5,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Generate a fractional Brownian surface using power-law spectrum.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    beta : float
        Power-law exponent (higher = smoother terrain)
    rng : Generator
        Random number generator
        
    Returns
    -------
    NDArray
        2D surface with power-law spectrum
    """
    if rng is None:
        rng = np.random.default_rng()
    
    fx, fy = _fft2_freq_grid(nx, ny)
    freq_mag = np.sqrt(fx**2 + fy**2)
    freq_mag[0, 0] = 1.0  # Avoid division by zero
    
    # Power-law spectrum
    power = freq_mag ** (-beta / 2)
    power[0, 0] = 0.0  # Zero DC component
    
    # Random phase
    phase = rng.uniform(0, 2 * np.pi, (nx, ny))
    
    # Complex spectrum
    spectrum = power * np.exp(1j * phase)
    
    # Inverse FFT to get surface
    surface = np.real(np.fft.ifft2(spectrum))
    
    # Normalize to [0, 1]
    surface = (surface - surface.min()) / (surface.max() - surface.min() + 1e-10)
    
    return surface


def domain_warp(
    surface: NDArray[np.float64],
    warp_strength: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """
    Apply domain warping for micro-relief and natural texture.
    
    Parameters
    ----------
    surface : NDArray
        Input surface
    warp_strength : float
        Strength of coordinate distortion
    rng : Generator
        Random number generator
        
    Returns
    -------
    NDArray
        Warped surface
    """
    if rng is None:
        rng = np.random.default_rng()
    
    nx, ny = surface.shape
    
    # Generate warp fields
    warp_x = fractional_surface(nx, ny, beta=3.0, rng=rng) - 0.5
    warp_y = fractional_surface(nx, ny, beta=3.0, rng=rng) - 0.5
    
    # Create coordinate grids
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Apply warp
    xx_warped = xx + warp_strength * nx * warp_x
    yy_warped = yy + warp_strength * ny * warp_y
    
    # Clip to valid range
    xx_warped = np.clip(xx_warped, 0, nx - 1)
    yy_warped = np.clip(yy_warped, 0, ny - 1)
    
    # Bilinear interpolation
    x0 = xx_warped.astype(int)
    y0 = yy_warped.astype(int)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    
    dx = xx_warped - x0
    dy = yy_warped - y0
    
    warped = (
        surface[x0, y0] * (1 - dx) * (1 - dy) +
        surface[x1, y0] * dx * (1 - dy) +
        surface[x0, y1] * (1 - dx) * dy +
        surface[x1, y1] * dx * dy
    )
    
    return warped


def ridged_mix(
    surface: NDArray[np.float64],
    ridge_power: float = 2.0,
) -> NDArray[np.float64]:
    """
    Apply ridge/valley sharpening transformation.
    
    Parameters
    ----------
    surface : NDArray
        Input surface (normalized to [0, 1])
    ridge_power : float
        Power for ridge transformation
        
    Returns
    -------
    NDArray
        Surface with enhanced ridges and valleys
    """
    # Center around 0.5
    centered = surface - 0.5
    
    # Apply signed power transformation
    ridged = np.sign(centered) * np.abs(centered) ** ridge_power
    
    # Normalize back to [0, 1]
    ridged = (ridged - ridged.min()) / (ridged.max() - ridged.min() + 1e-10)
    
    return ridged


def lowpass2d(
    surface: NDArray[np.float64],
    cutoff: float = 0.1,
) -> NDArray[np.float64]:
    """
    Apply 2D lowpass filter to surface.
    
    Parameters
    ----------
    surface : NDArray
        Input surface
    cutoff : float
        Cutoff frequency (fraction of Nyquist)
        
    Returns
    -------
    NDArray
        Smoothed surface
    """
    nx, ny = surface.shape
    fx, fy = _fft2_freq_grid(nx, ny)
    freq_mag = np.sqrt(fx**2 + fy**2)
    
    # Create lowpass filter
    filt = np.exp(-(freq_mag / cutoff) ** 4)
    
    # Apply filter
    spectrum = np.fft.fft2(surface)
    filtered = np.real(np.fft.ifft2(spectrum * filt))
    
    return filtered


def gaussian_blur(
    surface: NDArray[np.float64],
    sigma: float = 2.0,
) -> NDArray[np.float64]:
    """
    Apply Gaussian blur to surface.
    
    Parameters
    ----------
    surface : NDArray
        Input surface
    sigma : float
        Standard deviation of Gaussian kernel
        
    Returns
    -------
    NDArray
        Blurred surface
    """
    nx, ny = surface.shape
    fx, fy = _fft2_freq_grid(nx, ny)
    freq_mag_sq = fx**2 + fy**2
    
    # Gaussian filter in frequency domain
    filt = np.exp(-2 * (np.pi * sigma) ** 2 * freq_mag_sq)
    
    # Apply filter
    spectrum = np.fft.fft2(surface)
    blurred = np.real(np.fft.ifft2(spectrum * filt))
    
    return blurred


def quantum_seeded_topography(
    nx: int = 256,
    ny: int = 256,
    elevation_min: float = 0.0,
    elevation_max: float = 1000.0,
    beta: float = 2.5,
    warp_strength: float = 0.05,
    ridge_power: float = 1.5,
    smooth_sigma: float = 2.0,
    qrng: Optional[_QuantumRNGWrapper] = None,
) -> NDArray[np.float64]:
    """
    Generate quantum-seeded topography with realistic features.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    elevation_min, elevation_max : float
        Elevation range in meters
    beta : float
        Power-law exponent for terrain roughness
    warp_strength : float
        Strength of domain warping
    ridge_power : float
        Power for ridge enhancement
    smooth_sigma : float
        Gaussian smoothing sigma
    qrng : _QuantumRNGWrapper
        Quantum RNG wrapper
        
    Returns
    -------
    NDArray
        2D elevation field in meters
    """
    if qrng is None:
        qrng = _QuantumRNGWrapper()
    
    rng = qrng.base_rng
    
    # Generate base fractional surface
    surface = fractional_surface(nx, ny, beta=beta, rng=rng)
    
    # Add multi-scale detail
    detail1 = fractional_surface(nx, ny, beta=beta + 0.5, rng=rng)
    detail2 = fractional_surface(nx, ny, beta=beta + 1.0, rng=rng)
    
    surface = 0.6 * surface + 0.25 * detail1 + 0.15 * detail2
    
    # Apply domain warping
    surface = domain_warp(surface, warp_strength=warp_strength, rng=rng)
    
    # Apply ridge enhancement
    surface = ridged_mix(surface, ridge_power=ridge_power)
    
    # Smooth
    surface = gaussian_blur(surface, sigma=smooth_sigma)
    
    # Scale to elevation range
    elevation = elevation_min + (elevation_max - elevation_min) * surface
    
    return elevation


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _box_blur(arr: NDArray[np.float64], kernel_size: int = 3) -> NDArray[np.float64]:
    """Apply box blur using convolution."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    # Pad array
    pad = kernel_size // 2
    padded = np.pad(arr, pad, mode='reflect')
    
    # Convolve using FFT for efficiency
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i, j] = np.sum(
                padded[i:i+kernel_size, j:j+kernel_size] * kernel
            )
    
    return result


def _normalize(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize array to [0, 1] range."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-10:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


# =============================================================================
# TOPOGRAPHIC FIELD COMPUTATION
# =============================================================================

def compute_topo_fields(
    elevation: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute topographic derivative fields.
    
    Parameters
    ----------
    elevation : NDArray
        2D elevation field
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    Dict containing:
        - slope_x: East-West slope
        - slope_y: North-South slope
        - slope_magnitude: Total slope magnitude
        - aspect: Slope aspect angle (radians, 0=East, CCW positive)
        - curvature: Plan curvature
        - laplacian: Terrain laplacian
    """
    # Compute gradients
    slope_y, slope_x = np.gradient(elevation, dy, dx)
    
    # Slope magnitude
    slope_magnitude = np.sqrt(slope_x**2 + slope_y**2)
    
    # Aspect (direction of steepest descent)
    aspect = np.arctan2(-slope_y, -slope_x)
    
    # Second derivatives for curvature
    slope_xx = np.gradient(slope_x, dx, axis=1)
    slope_yy = np.gradient(slope_y, dy, axis=0)
    slope_xy = np.gradient(slope_x, dy, axis=0)
    
    # Plan curvature
    denom = (slope_x**2 + slope_y**2) ** 1.5 + 1e-10
    curvature = (
        slope_xx * slope_y**2 
        - 2 * slope_xy * slope_x * slope_y 
        + slope_yy * slope_x**2
    ) / denom
    
    # Laplacian
    laplacian = slope_xx + slope_yy
    
    return {
        'slope_x': slope_x,
        'slope_y': slope_y,
        'slope_magnitude': slope_magnitude,
        'aspect': aspect,
        'curvature': curvature,
        'laplacian': laplacian,
    }


def classify_windward_leeward(
    topo_fields: Dict[str, NDArray[np.float64]],
    wind_direction: float = 270.0,  # degrees, meteorological (270 = from west)
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Classify terrain as windward or leeward based on aspect and wind direction.
    
    Parameters
    ----------
    topo_fields : Dict
        Output from compute_topo_fields
    wind_direction : float
        Wind FROM direction in degrees (meteorological convention)
        
    Returns
    -------
    Tuple of (windward_mask, leeward_mask)
    """
    # Convert to radians and to direction wind is GOING (math convention)
    wind_to_rad = np.deg2rad((wind_direction + 180) % 360)
    
    aspect = topo_fields['aspect']
    slope_mag = topo_fields['slope_magnitude']
    
    # Calculate angle between slope face and wind direction
    angle_diff = np.abs(np.mod(aspect - wind_to_rad + np.pi, 2 * np.pi) - np.pi)
    
    # Windward: slope faces into wind (angle_diff > 90 degrees)
    # Leeward: slope faces away from wind (angle_diff < 90 degrees)
    min_slope = 0.01  # Minimum slope to classify
    
    windward = (angle_diff > np.pi / 2) & (slope_mag > min_slope)
    leeward = (angle_diff < np.pi / 2) & (slope_mag > min_slope)
    
    return windward, leeward


def classify_wind_barriers(
    elevation: NDArray[np.float64],
    topo_fields: Dict[str, NDArray[np.float64]],
    min_prominence: float = 50.0,
    min_slope: float = 0.1,
) -> NDArray[np.bool_]:
    """
    Identify wind barriers (ridges, hills that block wind).
    
    Parameters
    ----------
    elevation : NDArray
        Elevation field
    topo_fields : Dict
        Topographic fields
    min_prominence : float
        Minimum local prominence to be considered a barrier
    min_slope : float
        Minimum slope
        
    Returns
    -------
    NDArray[bool]
        Wind barrier mask
    """
    # Find local maxima
    blurred = gaussian_blur(elevation, sigma=5.0)
    local_max = elevation > blurred
    
    # Check prominence
    global_min = elevation.min()
    prominence = elevation - global_min
    high_enough = prominence > min_prominence
    
    # Check slope on flanks
    steep_enough = topo_fields['slope_magnitude'] > min_slope
    
    # Dilate the maxima to include barrier flanks
    barrier = local_max & high_enough
    
    # Include steep slopes near barriers
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(barrier, iterations=3)
    
    return dilated & steep_enough


def classify_wind_channels(
    elevation: NDArray[np.float64],
    topo_fields: Dict[str, NDArray[np.float64]],
    channel_depth: float = 20.0,
) -> NDArray[np.bool_]:
    """
    Identify wind channels (valleys, gaps that funnel wind).
    
    Parameters
    ----------
    elevation : NDArray
        Elevation field
    topo_fields : Dict
        Topographic fields
    channel_depth : float
        Minimum depth relative to surroundings
        
    Returns
    -------
    NDArray[bool]
        Wind channel mask
    """
    # Find concave areas (negative curvature = valleys)
    concave = topo_fields['curvature'] < -0.001
    
    # Check if locally low
    blurred = gaussian_blur(elevation, sigma=10.0)
    locally_low = elevation < blurred - channel_depth / 2
    
    return concave & locally_low


def classify_basins(
    elevation: NDArray[np.float64],
    topo_fields: Dict[str, NDArray[np.float64]],
    min_depth: float = 30.0,
    min_area_cells: int = 100,
) -> NDArray[np.int32]:
    """
    Identify and label enclosed basins.
    
    Parameters
    ----------
    elevation : NDArray
        Elevation field
    topo_fields : Dict
        Topographic fields
    min_depth : float
        Minimum basin depth
    min_area_cells : int
        Minimum basin area in grid cells
        
    Returns
    -------
    NDArray[int32]
        Basin labels (0 = no basin, 1+ = basin ID)
    """
    from scipy.ndimage import label
    
    # Find local minima regions
    blurred = gaussian_blur(elevation, sigma=8.0)
    basin_candidate = elevation < blurred - min_depth / 2
    
    # Also check for flat areas with surrounding higher terrain
    flat = topo_fields['slope_magnitude'] < 0.02
    
    # Combine criteria
    potential_basin = basin_candidate | (flat & (elevation < blurred))
    
    # Label connected components
    labeled, num_features = label(potential_basin)
    
    # Filter by size
    result = np.zeros_like(labeled)
    new_label = 1
    for i in range(1, num_features + 1):
        mask = labeled == i
        if np.sum(mask) >= min_area_cells:
            result[mask] = new_label
            new_label += 1
    
    return result


def extract_region_summaries(
    labeled: NDArray[np.int32],
    elevation: NDArray[np.float64],
) -> List[Dict[str, Any]]:
    """
    Extract summary statistics for labeled regions.
    
    Parameters
    ----------
    labeled : NDArray[int32]
        Region labels
    elevation : NDArray
        Elevation field
        
    Returns
    -------
    List of dicts with region properties
    """
    summaries = []
    unique_labels = np.unique(labeled)
    
    for lbl in unique_labels:
        if lbl == 0:
            continue
        
        mask = labeled == lbl
        region_elev = elevation[mask]
        
        # Find centroid
        coords = np.argwhere(mask)
        centroid = coords.mean(axis=0)
        
        summaries.append({
            'label': int(lbl),
            'area_cells': int(np.sum(mask)),
            'centroid_i': float(centroid[0]),
            'centroid_j': float(centroid[1]),
            'min_elevation': float(region_elev.min()),
            'max_elevation': float(region_elev.max()),
            'mean_elevation': float(region_elev.mean()),
        })
    
    return summaries


@dataclass
class WindStructures:
    """Container for wind-relevant terrain structures."""
    windward_mask: NDArray[np.bool_]
    leeward_mask: NDArray[np.bool_]
    barrier_mask: NDArray[np.bool_]
    channel_mask: NDArray[np.bool_]
    basin_labels: NDArray[np.int32]
    basin_summaries: List[Dict[str, Any]]
    wind_direction: float  # degrees


def build_wind_structures(
    elevation: NDArray[np.float64],
    wind_direction: float = 270.0,
    dx: float = 1.0,
    dy: float = 1.0,
) -> WindStructures:
    """
    Build all wind-relevant terrain structures.
    
    Parameters
    ----------
    elevation : NDArray
        Elevation field
    wind_direction : float
        Wind FROM direction in degrees
    dx, dy : float
        Grid spacing
        
    Returns
    -------
    WindStructures
        Container with all wind structure data
    """
    topo_fields = compute_topo_fields(elevation, dx, dy)
    
    windward, leeward = classify_windward_leeward(topo_fields, wind_direction)
    barriers = classify_wind_barriers(elevation, topo_fields)
    channels = classify_wind_channels(elevation, topo_fields)
    basin_labels = classify_basins(elevation, topo_fields)
    basin_summaries = extract_region_summaries(basin_labels, elevation)
    
    return WindStructures(
        windward_mask=windward,
        leeward_mask=leeward,
        barrier_mask=barriers,
        channel_mask=channels,
        basin_labels=basin_labels,
        basin_summaries=basin_summaries,
        wind_direction=wind_direction,
    )


# =============================================================================
# OROGRAPHIC LOW PRESSURE COMPUTATION
# =============================================================================

def compute_orographic_low_pressure(
    elevation: NDArray[np.float64],
    wind_structures: WindStructures,
    max_pressure_drop: float = 0.1,
) -> NDArray[np.float64]:
    """
    Compute orographic low-pressure likelihood map.
    
    Areas on leeward slopes and in basins have higher low-pressure likelihood,
    which affects storm behavior.
    
    Parameters
    ----------
    elevation : NDArray
        Elevation field
    wind_structures : WindStructures
        Wind structure data
    max_pressure_drop : float
        Maximum pressure drop factor (0-1)
        
    Returns
    -------
    NDArray
        Low-pressure likelihood (0 = normal, 1 = maximum low)
    """
    nx, ny = elevation.shape
    low_pressure = np.zeros((nx, ny), dtype=np.float64)
    
    # Leeward slopes create low pressure
    low_pressure[wind_structures.leeward_mask] += 0.4
    
    # Basins trap air and can create low pressure
    basin_contribution = (wind_structures.basin_labels > 0).astype(np.float64) * 0.3
    low_pressure += basin_contribution
    
    # Wind channels can create localized low pressure (Venturi effect)
    low_pressure[wind_structures.channel_mask] += 0.2
    
    # Normalize
    low_pressure = np.clip(low_pressure, 0, 1) * max_pressure_drop
    
    # Smooth
    low_pressure = gaussian_blur(low_pressure, sigma=3.0)
    
    return low_pressure


# =============================================================================
# STRATIGRAPHY MODULE
# =============================================================================

class LayerType(Enum):
    """Geological layer types."""
    TOPSOIL = auto()
    SUBSOIL = auto()
    COLLUVIUM = auto()
    SAPROLITE = auto()
    WEATHERED_BR = auto()  # Weathered bedrock
    SANDSTONE = auto()
    SHALE = auto()
    LIMESTONE = auto()
    BASEMENT = auto()
    BASEMENT_FLOOR = auto()


@dataclass
class MaterialProperties:
    """Material properties for a geological layer."""
    name: str
    density: float  # kg/m³
    porosity: float  # 0-1
    permeability: float  # m²
    color: Tuple[float, float, float]  # RGB for plotting


# Default material properties
MATERIAL_PROPERTIES: Dict[LayerType, MaterialProperties] = {
    LayerType.TOPSOIL: MaterialProperties(
        "Topsoil", 1200.0, 0.45, 1e-12, (0.4, 0.26, 0.13)
    ),
    LayerType.SUBSOIL: MaterialProperties(
        "Subsoil", 1500.0, 0.35, 1e-13, (0.6, 0.4, 0.2)
    ),
    LayerType.COLLUVIUM: MaterialProperties(
        "Colluvium", 1600.0, 0.30, 1e-13, (0.7, 0.5, 0.3)
    ),
    LayerType.SAPROLITE: MaterialProperties(
        "Saprolite", 1800.0, 0.25, 1e-14, (0.8, 0.6, 0.4)
    ),
    LayerType.WEATHERED_BR: MaterialProperties(
        "Weathered Bedrock", 2200.0, 0.15, 1e-15, (0.6, 0.6, 0.5)
    ),
    LayerType.SANDSTONE: MaterialProperties(
        "Sandstone", 2400.0, 0.20, 1e-14, (0.9, 0.8, 0.5)
    ),
    LayerType.SHALE: MaterialProperties(
        "Shale", 2500.0, 0.10, 1e-18, (0.4, 0.4, 0.4)
    ),
    LayerType.LIMESTONE: MaterialProperties(
        "Limestone", 2600.0, 0.15, 1e-16, (0.85, 0.85, 0.75)
    ),
    LayerType.BASEMENT: MaterialProperties(
        "Basement", 2700.0, 0.05, 1e-19, (0.5, 0.5, 0.5)
    ),
    LayerType.BASEMENT_FLOOR: MaterialProperties(
        "Basement Floor", 2800.0, 0.01, 1e-20, (0.3, 0.3, 0.3)
    ),
}


def soil_thickness_from_slope(
    slope: NDArray[np.float64],
    max_thickness: float = 2.0,
    slope_factor: float = 5.0,
) -> NDArray[np.float64]:
    """
    Calculate soil thickness based on slope (thinner on steep slopes).
    
    Parameters
    ----------
    slope : NDArray
        Slope magnitude
    max_thickness : float
        Maximum soil thickness (meters)
    slope_factor : float
        Controls sensitivity to slope
        
    Returns
    -------
    NDArray
        Soil thickness field
    """
    return max_thickness * np.exp(-slope_factor * slope)


def colluvium_thickness_field(
    elevation: NDArray[np.float64],
    slope: NDArray[np.float64],
    max_thickness: float = 5.0,
) -> NDArray[np.float64]:
    """Calculate colluvium thickness (accumulates at slope bases)."""
    # Colluvium accumulates where slope decreases (deposition zones)
    slope_gradient = np.gradient(slope, axis=0) + np.gradient(slope, axis=1)
    deposition = np.maximum(-slope_gradient, 0)
    
    # Normalize and scale
    deposition = _normalize(deposition)
    return max_thickness * deposition


def saprolite_thickness_field(
    elevation: NDArray[np.float64],
    rng: np.random.Generator,
    mean_thickness: float = 10.0,
    std_thickness: float = 3.0,
) -> NDArray[np.float64]:
    """Calculate saprolite thickness (weathered rock)."""
    nx, ny = elevation.shape
    base = mean_thickness + std_thickness * rng.standard_normal((nx, ny))
    base = np.maximum(base, 0)
    
    # Thicker in low areas (more weathering time)
    elev_norm = _normalize(elevation)
    return base * (1.5 - elev_norm)


def weathered_rind_thickness_field(
    elevation: NDArray[np.float64],
    rng: np.random.Generator,
    mean_thickness: float = 3.0,
) -> NDArray[np.float64]:
    """Calculate weathered bedrock rind thickness."""
    nx, ny = elevation.shape
    return mean_thickness * (0.5 + 0.5 * rng.random((nx, ny)))


def till_thickness_field(
    elevation: NDArray[np.float64],
    glaciated_fraction: float = 0.3,
    max_thickness: float = 20.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Calculate glacial till thickness (if glaciated)."""
    if rng is None:
        rng = np.random.default_rng()
    
    nx, ny = elevation.shape
    
    # Till primarily in valleys
    elev_norm = _normalize(elevation)
    valley_factor = 1 - elev_norm
    
    # Random presence
    presence = rng.random((nx, ny)) < glaciated_fraction
    
    thickness = max_thickness * valley_factor * presence
    return gaussian_blur(thickness, sigma=5.0)


def loess_thickness_field(
    elevation: NDArray[np.float64],
    wind_direction: float = 270.0,
    max_thickness: float = 5.0,
) -> NDArray[np.float64]:
    """Calculate loess (wind-deposited silt) thickness."""
    nx, ny = elevation.shape
    topo_fields = compute_topo_fields(elevation)
    
    # Loess accumulates on leeward slopes
    _, leeward = classify_windward_leeward(topo_fields, wind_direction)
    
    thickness = max_thickness * leeward.astype(np.float64)
    return gaussian_blur(thickness, sigma=3.0)


def dune_thickness_field(
    elevation: NDArray[np.float64],
    slope: NDArray[np.float64],
    max_thickness: float = 10.0,
    slope_threshold: float = 0.05,
) -> NDArray[np.float64]:
    """Calculate dune sand thickness (in flat, low areas)."""
    elev_norm = _normalize(elevation)
    
    # Dunes form in flat, low areas
    flat = slope < slope_threshold
    low = elev_norm < 0.3
    
    thickness = max_thickness * flat * low
    return gaussian_blur(thickness, sigma=2.0)


def crust_thickness_field(
    nx: int,
    ny: int,
    base_thickness: float = 30000.0,
    variation: float = 5000.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Calculate crustal thickness variation."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Long-wavelength variation
    crust = fractional_surface(nx, ny, beta=4.0, rng=rng)
    crust = base_thickness + variation * (2 * crust - 1)
    
    return crust


@dataclass
class StratigraphicColumn:
    """Stratigraphy at a single grid point."""
    layer_types: List[LayerType]
    layer_tops: List[float]  # Elevation of layer tops
    layer_bottoms: List[float]  # Elevation of layer bottoms
    layer_thicknesses: List[float]


@dataclass
class Stratigraphy:
    """Full 3D stratigraphy."""
    nx: int
    ny: int
    surface_elevation: NDArray[np.float64]
    layer_thickness: Dict[LayerType, NDArray[np.float64]]
    layer_top_elevation: Dict[LayerType, NDArray[np.float64]]
    layer_order: List[LayerType]  # Top to bottom
    top_material_map: NDArray[np.int32]  # Top material at surface
    top_facies_map: NDArray[np.int32]  # Top facies (simplified grouping)


def compute_top_material_map(
    layer_thickness: Dict[LayerType, NDArray[np.float64]],
    layer_order: List[LayerType],
) -> NDArray[np.int32]:
    """Determine the top material at each grid point."""
    nx, ny = next(iter(layer_thickness.values())).shape
    result = np.zeros((nx, ny), dtype=np.int32)
    
    for i, layer_type in enumerate(layer_order):
        thickness = layer_thickness[layer_type]
        mask = (result == 0) & (thickness > 0.1)
        result[mask] = layer_type.value
    
    return result


def compute_top_facies_map(
    top_material_map: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Group materials into broader facies categories."""
    facies = np.zeros_like(top_material_map)
    
    # Soil facies (1)
    soil_types = [LayerType.TOPSOIL.value, LayerType.SUBSOIL.value]
    facies[np.isin(top_material_map, soil_types)] = 1
    
    # Unconsolidated sediment facies (2)
    sediment_types = [LayerType.COLLUVIUM.value, LayerType.SAPROLITE.value]
    facies[np.isin(top_material_map, sediment_types)] = 2
    
    # Weathered rock facies (3)
    weathered_types = [LayerType.WEATHERED_BR.value]
    facies[np.isin(top_material_map, weathered_types)] = 3
    
    # Sedimentary rock facies (4)
    sed_rock_types = [LayerType.SANDSTONE.value, LayerType.SHALE.value, 
                      LayerType.LIMESTONE.value]
    facies[np.isin(top_material_map, sed_rock_types)] = 4
    
    # Basement facies (5)
    basement_types = [LayerType.BASEMENT.value, LayerType.BASEMENT_FLOOR.value]
    facies[np.isin(top_material_map, basement_types)] = 5
    
    return facies


def generate_stratigraphy(
    elevation: NDArray[np.float64],
    rng: Optional[np.random.Generator] = None,
    include_glacial: bool = False,
    include_loess: bool = True,
) -> Stratigraphy:
    """
    Generate full stratigraphy based on terrain.
    
    Parameters
    ----------
    elevation : NDArray
        Surface elevation field
    rng : Generator
        Random number generator
    include_glacial : bool
        Include glacial deposits
    include_loess : bool
        Include wind-deposited loess
        
    Returns
    -------
    Stratigraphy
        Complete stratigraphic model
    """
    if rng is None:
        rng = np.random.default_rng()
    
    nx, ny = elevation.shape
    topo_fields = compute_topo_fields(elevation)
    slope = topo_fields['slope_magnitude']
    
    # Calculate layer thicknesses
    layer_thickness: Dict[LayerType, NDArray[np.float64]] = {}
    
    # Surface soils
    layer_thickness[LayerType.TOPSOIL] = soil_thickness_from_slope(
        slope, max_thickness=0.5
    )
    layer_thickness[LayerType.SUBSOIL] = soil_thickness_from_slope(
        slope, max_thickness=1.5
    )
    
    # Unconsolidated deposits
    layer_thickness[LayerType.COLLUVIUM] = colluvium_thickness_field(
        elevation, slope
    )
    
    if include_loess:
        loess = loess_thickness_field(elevation)
        layer_thickness[LayerType.COLLUVIUM] += loess
    
    if include_glacial:
        till = till_thickness_field(elevation, rng=rng)
        layer_thickness[LayerType.COLLUVIUM] += till
    
    # Weathering profile
    layer_thickness[LayerType.SAPROLITE] = saprolite_thickness_field(
        elevation, rng
    )
    layer_thickness[LayerType.WEATHERED_BR] = weathered_rind_thickness_field(
        elevation, rng
    )
    
    # Bedrock layers (simplified)
    # Create variable-thickness sedimentary sequence
    sed_base = fractional_surface(nx, ny, beta=3.0, rng=rng)
    
    layer_thickness[LayerType.SANDSTONE] = 50.0 * (0.5 + sed_base)
    layer_thickness[LayerType.SHALE] = 30.0 * (0.3 + 0.7 * (1 - sed_base))
    layer_thickness[LayerType.LIMESTONE] = 40.0 * (0.2 + 0.8 * sed_base)
    
    # Basement
    layer_thickness[LayerType.BASEMENT] = crust_thickness_field(nx, ny, rng=rng)
    layer_thickness[LayerType.BASEMENT_FLOOR] = np.full((nx, ny), 1000.0)
    
    # Define layer order (top to bottom)
    layer_order = [
        LayerType.TOPSOIL,
        LayerType.SUBSOIL,
        LayerType.COLLUVIUM,
        LayerType.SAPROLITE,
        LayerType.WEATHERED_BR,
        LayerType.SANDSTONE,
        LayerType.SHALE,
        LayerType.LIMESTONE,
        LayerType.BASEMENT,
        LayerType.BASEMENT_FLOOR,
    ]
    
    # Calculate layer top elevations
    layer_top_elevation: Dict[LayerType, NDArray[np.float64]] = {}
    current_top = elevation.copy()
    
    for layer_type in layer_order:
        layer_top_elevation[layer_type] = current_top.copy()
        current_top = current_top - layer_thickness[layer_type]
    
    # Compute surface material maps
    top_material_map = compute_top_material_map(layer_thickness, layer_order)
    top_facies_map = compute_top_facies_map(top_material_map)
    
    return Stratigraphy(
        nx=nx,
        ny=ny,
        surface_elevation=elevation,
        layer_thickness=layer_thickness,
        layer_top_elevation=layer_top_elevation,
        layer_order=layer_order,
        top_material_map=top_material_map,
        top_facies_map=top_facies_map,
    )


# =============================================================================
# VISUALIZATION - CROSS SECTIONS
# =============================================================================

def plot_cross_section(
    stratigraphy: Stratigraphy,
    i_slice: Optional[int] = None,
    j_slice: Optional[int] = None,
    depth_limit: float = 200.0,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a stratigraphic cross-section.
    
    Parameters
    ----------
    stratigraphy : Stratigraphy
        Stratigraphic model
    i_slice : int, optional
        Row index for E-W cross-section
    j_slice : int, optional
        Column index for N-S cross-section
    depth_limit : float
        Maximum depth below surface to show
    ax : Axes, optional
        Matplotlib axes
        
    Returns
    -------
    Axes
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    if i_slice is not None:
        # E-W cross-section
        surface = stratigraphy.surface_elevation[i_slice, :]
        x = np.arange(stratigraphy.ny)
        direction = 'E-W'
    elif j_slice is not None:
        # N-S cross-section
        surface = stratigraphy.surface_elevation[:, j_slice]
        x = np.arange(stratigraphy.nx)
        direction = 'N-S'
    else:
        # Default to middle E-W
        i_slice = stratigraphy.nx // 2
        surface = stratigraphy.surface_elevation[i_slice, :]
        x = np.arange(stratigraphy.ny)
        direction = 'E-W'
    
    # Plot each layer
    prev_bottom = surface.copy()
    
    for layer_type in stratigraphy.layer_order:
        if i_slice is not None:
            thickness = stratigraphy.layer_thickness[layer_type][i_slice, :]
        else:
            thickness = stratigraphy.layer_thickness[layer_type][:, j_slice]
        
        bottom = prev_bottom - thickness
        
        # Clip to depth limit
        min_elev = surface - depth_limit
        bottom_clipped = np.maximum(bottom, min_elev)
        prev_clipped = np.maximum(prev_bottom, min_elev)
        
        # Get material color
        props = MATERIAL_PROPERTIES[layer_type]
        
        ax.fill_between(
            x, prev_clipped, bottom_clipped,
            color=props.color, label=props.name,
            alpha=0.8
        )
        
        prev_bottom = bottom
    
    # Plot surface
    ax.plot(x, surface, 'k-', linewidth=1.5, label='Surface')
    
    ax.set_xlabel(f'Distance ({direction})')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(f'Stratigraphic Cross-Section ({direction})')
    ax.legend(loc='upper right', fontsize=8)
    
    return ax


def plot_cross_sections_xy(
    stratigraphy: Stratigraphy,
    depth_limit: float = 200.0,
) -> plt.Figure:
    """Plot both E-W and N-S cross-sections."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    plot_cross_section(stratigraphy, i_slice=stratigraphy.nx // 2,
                       depth_limit=depth_limit, ax=axes[0])
    plot_cross_section(stratigraphy, j_slice=stratigraphy.ny // 2,
                       depth_limit=depth_limit, ax=axes[1])
    
    plt.tight_layout()
    return fig


def plot_wind_structures_debug(
    elevation: NDArray[np.float64],
    wind_structures: WindStructures,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Debug plot showing all wind-relevant structures.
    
    Parameters
    ----------
    elevation : NDArray
        Elevation field
    wind_structures : WindStructures
        Wind structure data
    figsize : tuple
        Figure size
        
    Returns
    -------
    Figure
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Elevation
    im0 = axes[0, 0].imshow(elevation.T, origin='lower', cmap='terrain')
    axes[0, 0].set_title('Elevation')
    plt.colorbar(im0, ax=axes[0, 0], label='m')
    
    # Windward/Leeward
    combined = np.zeros_like(elevation)
    combined[wind_structures.windward_mask] = 1
    combined[wind_structures.leeward_mask] = -1
    im1 = axes[0, 1].imshow(combined.T, origin='lower', cmap='RdBu', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'Windward/Leeward (wind from {wind_structures.wind_direction}°)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Wind barriers
    axes[0, 2].imshow(elevation.T, origin='lower', cmap='terrain', alpha=0.5)
    axes[0, 2].imshow(wind_structures.barrier_mask.T, origin='lower', 
                       cmap='Reds', alpha=0.5)
    axes[0, 2].set_title('Wind Barriers')
    
    # Wind channels
    axes[1, 0].imshow(elevation.T, origin='lower', cmap='terrain', alpha=0.5)
    axes[1, 0].imshow(wind_structures.channel_mask.T, origin='lower',
                       cmap='Blues', alpha=0.5)
    axes[1, 0].set_title('Wind Channels')
    
    # Basins
    basin_masked = np.ma.masked_where(
        wind_structures.basin_labels == 0,
        wind_structures.basin_labels
    )
    axes[1, 1].imshow(elevation.T, origin='lower', cmap='terrain', alpha=0.5)
    axes[1, 1].imshow(basin_masked.T, origin='lower', cmap='tab20')
    axes[1, 1].set_title(f'Basins ({len(wind_structures.basin_summaries)} found)')
    
    # Composite
    composite = np.zeros((*elevation.shape, 3))
    composite[..., 0] = wind_structures.barrier_mask * 0.8
    composite[..., 1] = wind_structures.channel_mask * 0.8
    composite[..., 2] = (wind_structures.basin_labels > 0) * 0.8
    axes[1, 2].imshow(np.transpose(composite, (1, 0, 2)), origin='lower')
    axes[1, 2].set_title('Composite (R=barriers, G=channels, B=basins)')
    
    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig


# =============================================================================
# CLIMATE/WEATHER TREND MODULE
# =============================================================================

@dataclass
class ClimateTrend:
    """Climate parameters for a given year."""
    year: int
    storm_frequency: float  # storms per year
    mean_storm_duration: float  # hours
    mean_storm_intensity: float  # mm/hr
    annual_rainfall_target: float  # mm/year
    preferred_wind_direction: float  # degrees


def get_year_trend(
    year: int,
    base_year: int = 0,
    base_frequency: float = 50.0,
    base_duration: float = 6.0,
    base_intensity: float = 5.0,
    base_rainfall: float = 800.0,
    base_wind_dir: float = 270.0,
    trend_per_year: float = 0.01,  # fractional change per year
) -> ClimateTrend:
    """
    Get deterministic climate trend for a given year.
    
    Parameters
    ----------
    year : int
        Simulation year
    base_year : int
        Reference year
    base_* : float
        Base climate parameters
    trend_per_year : float
        Fractional change per year (positive = increasing)
        
    Returns
    -------
    ClimateTrend
        Climate parameters for the year
    """
    years_elapsed = year - base_year
    trend_factor = 1.0 + trend_per_year * years_elapsed
    
    return ClimateTrend(
        year=year,
        storm_frequency=base_frequency * trend_factor,
        mean_storm_duration=base_duration * (1 + 0.005 * years_elapsed),
        mean_storm_intensity=base_intensity * trend_factor,
        annual_rainfall_target=base_rainfall * trend_factor,
        preferred_wind_direction=base_wind_dir + 0.5 * years_elapsed,  # Slow drift
    )


def get_seasonal_modifiers(month: int) -> Dict[str, float]:
    """
    Get seasonal modification factors for climate parameters.
    
    Parameters
    ----------
    month : int
        Month (1-12)
        
    Returns
    -------
    Dict with modifiers for frequency, intensity, duration
    """
    # Simple sinusoidal pattern (wet season in winter)
    phase = 2 * np.pi * (month - 1) / 12
    
    return {
        'frequency': 1.0 + 0.5 * np.cos(phase),  # More storms in winter
        'intensity': 1.0 + 0.3 * np.cos(phase),
        'duration': 1.0 + 0.2 * np.cos(phase),
    }


def _terrain_climate_modifiers(
    elevation: NDArray[np.float64],
    wind_structures: WindStructures,
) -> Dict[str, float]:
    """
    Calculate terrain-based climate modifiers.
    
    Parameters
    ----------
    elevation : NDArray
        Elevation field
    wind_structures : WindStructures
        Wind structure data
        
    Returns
    -------
    Dict with terrain-based modifiers
    """
    # Calculate terrain statistics
    relief = elevation.max() - elevation.min()
    mean_elev = elevation.mean()
    
    barrier_fraction = wind_structures.barrier_mask.mean()
    channel_fraction = wind_structures.channel_mask.mean()
    basin_fraction = (wind_structures.basin_labels > 0).mean()
    
    return {
        'orographic_enhancement': 1.0 + 0.5 * (relief / 1000),
        'barrier_blocking': 1.0 - 0.2 * barrier_fraction,
        'channel_funneling': 1.0 + 0.3 * channel_fraction,
        'basin_pooling': 1.0 + 0.2 * basin_fraction,
        'elevation_effect': 1.0 - 0.1 * (mean_elev / 1000),
    }


# =============================================================================
# STORM SCHEDULING AND SAMPLING
# =============================================================================

def sample_interstorm_gap(
    mean_gap_hours: float,
    rng: np.random.Generator,
) -> float:
    """
    Sample inter-storm gap duration using exponential distribution.
    
    Parameters
    ----------
    mean_gap_hours : float
        Mean gap between storms
    rng : Generator
        Random number generator
        
    Returns
    -------
    float
        Gap duration in hours
    """
    return rng.exponential(mean_gap_hours)


def sample_storm_duration(
    mean_duration_hours: float,
    shape: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Sample storm duration using Weibull distribution.
    
    Parameters
    ----------
    mean_duration_hours : float
        Mean storm duration
    shape : float
        Weibull shape parameter (>1 = peaked distribution)
    rng : Generator
        Random number generator
        
    Returns
    -------
    float
        Storm duration in hours
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Convert mean to Weibull scale
    from scipy.special import gamma
    scale = mean_duration_hours / gamma(1 + 1/shape)
    
    return rng.weibull(shape) * scale


def _year_seed_from_quantum(
    year: int,
    qrng: _QuantumRNGWrapper,
) -> int:
    """Generate a quantum-based seed for a specific year."""
    # Use a small amount of quantum randomness to seed each year
    quantum_bits = qrng.integers(0, 2**31, size=1)[0]
    return int(quantum_bits) ^ (year * 2654435761)  # Mix with year


def _make_year_rng(
    year: int,
    qrng: _QuantumRNGWrapper,
) -> np.random.Generator:
    """Create a seeded RNG for a specific year."""
    seed = _year_seed_from_quantum(year, qrng)
    return np.random.default_rng(seed)


@dataclass
class StormEvent:
    """A single storm event."""
    storm_id: int
    year: int
    start_hour: float  # Hour of year (0-8760)
    duration_hours: float
    peak_intensity: float  # mm/hr
    wind_direction: float  # degrees
    wind_speed: float  # m/s
    center_x: float  # Normalized (0-1)
    center_y: float  # Normalized (0-1)
    motion_x: float  # Grid cells per hour
    motion_y: float  # Grid cells per hour


def generate_storm_schedule_for_year(
    year: int,
    climate: ClimateTrend,
    qrng: _QuantumRNGWrapper,
    hours_per_year: float = 8760.0,
) -> List[StormEvent]:
    """
    Generate storm schedule for a single year.
    
    Parameters
    ----------
    year : int
        Simulation year
    climate : ClimateTrend
        Climate parameters for the year
    qrng : _QuantumRNGWrapper
        Quantum RNG wrapper
    hours_per_year : float
        Hours in year
        
    Returns
    -------
    List[StormEvent]
        List of storm events for the year
    """
    # Create year-specific RNG for efficiency
    rng = _make_year_rng(year, qrng)
    
    storms = []
    current_hour = 0.0
    storm_id = 0
    
    # Mean gap between storms
    mean_gap = hours_per_year / climate.storm_frequency
    
    while current_hour < hours_per_year:
        # Sample gap to next storm
        gap = sample_interstorm_gap(mean_gap, rng)
        current_hour += gap
        
        if current_hour >= hours_per_year:
            break
        
        # Get seasonal modifiers
        month = int(current_hour / (hours_per_year / 12)) + 1
        month = min(max(month, 1), 12)
        seasonal = get_seasonal_modifiers(month)
        
        # Sample storm properties
        duration = sample_storm_duration(
            climate.mean_storm_duration * seasonal['duration'],
            rng=rng
        )
        
        # Intensity with some randomness
        intensity = climate.mean_storm_intensity * seasonal['intensity']
        intensity *= (0.5 + rng.random())
        
        # Wind direction with randomness around preferred
        wind_dir = climate.preferred_wind_direction + rng.normal(0, 30)
        wind_speed = 5.0 + 10.0 * rng.random()
        
        # Storm center and motion
        center_x = rng.random()
        center_y = rng.random()
        
        # Motion roughly aligned with wind
        wind_rad = np.deg2rad(wind_dir)
        base_motion = wind_speed * 0.1  # Scale down
        motion_x = base_motion * np.sin(wind_rad) + rng.normal(0, 0.5)
        motion_y = base_motion * np.cos(wind_rad) + rng.normal(0, 0.5)
        
        storms.append(StormEvent(
            storm_id=storm_id,
            year=year,
            start_hour=current_hour,
            duration_hours=duration,
            peak_intensity=intensity,
            wind_direction=wind_dir,
            wind_speed=wind_speed,
            center_x=center_x,
            center_y=center_y,
            motion_x=motion_x,
            motion_y=motion_y,
        ))
        
        storm_id += 1
        current_hour += duration
    
    return storms


# =============================================================================
# STORM WEATHER FIELD GENERATION
# =============================================================================

def _bilinear_sample_scalar(
    field: NDArray[np.float64],
    x: float,
    y: float,
) -> float:
    """Bilinear interpolation of a scalar field."""
    nx, ny = field.shape
    
    # Clamp to valid range
    x = max(0, min(x, nx - 1.001))
    y = max(0, min(y, ny - 1.001))
    
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    
    dx = x - x0
    dy = y - y0
    
    return (
        field[x0, y0] * (1 - dx) * (1 - dy) +
        field[x1, y0] * dx * (1 - dy) +
        field[x0, y1] * (1 - dx) * dy +
        field[x1, y1] * dx * dy
    )


def generate_storm_weather_fields(
    storm: StormEvent,
    elevation: NDArray[np.float64],
    wind_structures: WindStructures,
    orographic_low: NDArray[np.float64],
    time_steps: int = 24,
    qrng: Optional[_QuantumRNGWrapper] = None,
) -> NDArray[np.float64]:
    """
    Generate spatial-temporal rain intensity fields for a storm.
    
    Parameters
    ----------
    storm : StormEvent
        Storm parameters
    elevation : NDArray
        Elevation field
    wind_structures : WindStructures
        Wind structure data
    orographic_low : NDArray
        Orographic low-pressure field
    time_steps : int
        Number of time steps
    qrng : _QuantumRNGWrapper
        Quantum RNG
        
    Returns
    -------
    NDArray
        Rain intensity field (nx, ny, time_steps) in mm/hr
    """
    if qrng is None:
        qrng = _QuantumRNGWrapper()
    
    nx, ny = elevation.shape
    rng = qrng.base_rng
    
    # Initialize output
    rain = np.zeros((nx, ny, time_steps), dtype=np.float64)
    
    # Time array
    dt = storm.duration_hours / time_steps
    times = np.linspace(0, storm.duration_hours, time_steps)
    
    # Temporal intensity profile (rises then falls)
    time_norm = times / storm.duration_hours
    temporal_profile = 4 * time_norm * (1 - time_norm)  # Parabolic
    
    # Normalize elevation for orographic effects
    elev_norm = _normalize(elevation)
    
    for t_idx, t in enumerate(times):
        # Storm center position at this time
        cx = storm.center_x * nx + storm.motion_x * t
        cy = storm.center_y * ny + storm.motion_y * t
        
        # Distance from storm center
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        
        # Storm radius (varies with intensity)
        storm_radius = 30 + 20 * storm.peak_intensity / 10
        
        # Spatial intensity pattern
        spatial = np.exp(-dist**2 / (2 * storm_radius**2))
        
        # Orographic enhancement on windward slopes
        orographic_factor = 1.0 + 0.5 * wind_structures.windward_mask
        orographic_factor -= 0.3 * wind_structures.leeward_mask
        
        # Elevation effect
        elev_factor = 1.0 + 0.3 * elev_norm
        
        # Low-pressure enhancement
        low_pressure_factor = 1.0 + orographic_low
        
        # Combine factors
        intensity = (
            storm.peak_intensity 
            * temporal_profile[t_idx]
            * spatial
            * orographic_factor
            * elev_factor
            * low_pressure_factor
        )
        
        # Add some noise
        noise = 1.0 + 0.1 * rng.standard_normal((nx, ny))
        intensity *= np.maximum(noise, 0)
        
        rain[:, :, t_idx] = np.maximum(intensity, 0)
    
    return rain


# =============================================================================
# RAINFALL ACCUMULATION
# =============================================================================

def _infer_dt_hours(rain_field: NDArray[np.float64], storm_duration: float) -> float:
    """Infer time step from rain field shape."""
    n_times = rain_field.shape[2]
    return storm_duration / n_times


def accumulate_rain_for_storm(
    rain_field: NDArray[np.float64],
    storm_duration: float,
) -> NDArray[np.float64]:
    """
    Accumulate rainfall for a single storm.
    
    Parameters
    ----------
    rain_field : NDArray
        Rain intensity (nx, ny, time) in mm/hr
    storm_duration : float
        Storm duration in hours
        
    Returns
    -------
    NDArray
        Total rainfall depth (nx, ny) in mm
    """
    dt = _infer_dt_hours(rain_field, storm_duration)
    return np.sum(rain_field, axis=2) * dt


@dataclass
class RainfallAccumulation:
    """Accumulated rainfall data."""
    total_depth: NDArray[np.float64]  # mm
    hit_count: NDArray[np.int32]  # Number of storms hitting each cell
    per_year_depth: Optional[List[NDArray[np.float64]]] = None
    per_storm_depth: Optional[List[NDArray[np.float64]]] = None


def accumulate_rain_over_simulation(
    storms: List[StormEvent],
    rain_fields: List[NDArray[np.float64]],
    climate_targets: List[float],
    store_per_year: bool = False,
    store_per_storm: bool = False,
) -> RainfallAccumulation:
    """
    Accumulate rainfall over entire simulation.
    
    Parameters
    ----------
    storms : List[StormEvent]
        All storm events
    rain_fields : List[NDArray]
        Rain fields for each storm
    climate_targets : List[float]
        Annual rainfall targets for scaling
    store_per_year : bool
        Store per-year accumulations
    store_per_storm : bool
        Store per-storm accumulations
        
    Returns
    -------
    RainfallAccumulation
        Accumulated rainfall data
    """
    if len(storms) == 0:
        nx, ny = 256, 256  # Default
        return RainfallAccumulation(
            total_depth=np.zeros((nx, ny)),
            hit_count=np.zeros((nx, ny), dtype=np.int32),
        )
    
    nx, ny, _ = rain_fields[0].shape
    total_depth = np.zeros((nx, ny), dtype=np.float64)
    hit_count = np.zeros((nx, ny), dtype=np.int32)
    
    per_year_depth = [] if store_per_year else None
    per_storm_depth = [] if store_per_storm else None
    
    # Group storms by year
    years = sorted(set(s.year for s in storms))
    
    for year in years:
        year_storms = [(s, rf) for s, rf in zip(storms, rain_fields) if s.year == year]
        year_depth = np.zeros((nx, ny), dtype=np.float64)
        
        for storm, rain_field in year_storms:
            storm_depth = accumulate_rain_for_storm(rain_field, storm.duration_hours)
            year_depth += storm_depth
            hit_count += (storm_depth > 0.1).astype(np.int32)
            
            if store_per_storm:
                per_storm_depth.append(storm_depth)
        
        # Scale to match climate target if needed
        if year < len(climate_targets):
            target = climate_targets[year]
            current_mean = year_depth.mean()
            if current_mean > 0:
                scale = target / current_mean
                year_depth *= min(scale, 2.0)  # Cap scaling
        
        total_depth += year_depth
        
        if store_per_year:
            per_year_depth.append(year_depth)
    
    return RainfallAccumulation(
        total_depth=total_depth,
        hit_count=hit_count,
        per_year_depth=per_year_depth,
        per_storm_depth=per_storm_depth,
    )


# =============================================================================
# TOP-LEVEL SIMULATION DRIVER
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for multi-year weather simulation."""
    nx: int = 128
    ny: int = 128
    n_years: int = 10
    elevation_min: float = 0.0
    elevation_max: float = 500.0
    base_wind_direction: float = 270.0
    include_stratigraphy: bool = True
    store_per_year: bool = True
    store_per_storm: bool = False


@dataclass
class SimulationResults:
    """Results from multi-year weather simulation."""
    config: SimulationConfig
    elevation: NDArray[np.float64]
    stratigraphy: Optional[Stratigraphy]
    wind_structures: WindStructures
    orographic_low: NDArray[np.float64]
    storms: List[StormEvent]
    rainfall: RainfallAccumulation
    climate_trends: List[ClimateTrend]


def run_multi_year_weather_simulation(
    config: Optional[SimulationConfig] = None,
    qrng: Optional[_QuantumRNGWrapper] = None,
    verbose: bool = True,
) -> SimulationResults:
    """
    Run the complete multi-year weather simulation.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration
    qrng : _QuantumRNGWrapper
        Quantum RNG wrapper
    verbose : bool
        Print progress
        
    Returns
    -------
    SimulationResults
        Complete simulation results
    """
    if config is None:
        config = SimulationConfig()
    
    if qrng is None:
        qrng = _QuantumRNGWrapper()
    
    if verbose:
        print("=" * 60)
        print("QUANTUM-SEEDED TERRAIN AND WEATHER SIMULATION")
        print("=" * 60)
    
    # Generate terrain
    if verbose:
        print("\n[1/6] Generating quantum-seeded topography...")
    
    elevation = quantum_seeded_topography(
        nx=config.nx,
        ny=config.ny,
        elevation_min=config.elevation_min,
        elevation_max=config.elevation_max,
        qrng=qrng,
    )
    
    if verbose:
        print(f"  Elevation range: {elevation.min():.1f} - {elevation.max():.1f} m")
    
    # Generate stratigraphy
    stratigraphy = None
    if config.include_stratigraphy:
        if verbose:
            print("\n[2/6] Generating stratigraphy...")
        stratigraphy = generate_stratigraphy(elevation, rng=qrng.base_rng)
        if verbose:
            print(f"  Layers: {len(stratigraphy.layer_order)}")
    
    # Build wind structures
    if verbose:
        print("\n[3/6] Analyzing wind-relevant terrain structures...")
    
    wind_structures = build_wind_structures(
        elevation,
        wind_direction=config.base_wind_direction,
    )
    
    if verbose:
        print(f"  Windward area: {100 * wind_structures.windward_mask.mean():.1f}%")
        print(f"  Leeward area: {100 * wind_structures.leeward_mask.mean():.1f}%")
        print(f"  Wind barriers: {100 * wind_structures.barrier_mask.mean():.1f}%")
        print(f"  Wind channels: {100 * wind_structures.channel_mask.mean():.1f}%")
        print(f"  Basins found: {len(wind_structures.basin_summaries)}")
    
    # Compute orographic low pressure
    if verbose:
        print("\n[4/6] Computing orographic low-pressure field...")
    
    orographic_low = compute_orographic_low_pressure(elevation, wind_structures)
    
    # Generate storm schedule
    if verbose:
        print("\n[5/6] Generating storm schedule...")
    
    all_storms = []
    climate_trends = []
    
    for year in range(config.n_years):
        climate = get_year_trend(year)
        climate_trends.append(climate)
        
        year_storms = generate_storm_schedule_for_year(year, climate, qrng)
        all_storms.extend(year_storms)
        
        if verbose:
            print(f"  Year {year}: {len(year_storms)} storms")
    
    if verbose:
        print(f"  Total storms: {len(all_storms)}")
    
    # Generate rain fields and accumulate
    if verbose:
        print("\n[6/6] Generating storm weather fields and accumulating rainfall...")
    
    rain_fields = []
    for i, storm in enumerate(all_storms):
        rain_field = generate_storm_weather_fields(
            storm, elevation, wind_structures, orographic_low,
            qrng=qrng,
        )
        rain_fields.append(rain_field)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_storms)} storms")
    
    # Accumulate rainfall
    climate_targets = [ct.annual_rainfall_target for ct in climate_trends]
    rainfall = accumulate_rain_over_simulation(
        all_storms,
        rain_fields,
        climate_targets,
        store_per_year=config.store_per_year,
        store_per_storm=config.store_per_storm,
    )
    
    if verbose:
        print(f"\n  Mean annual rainfall: {rainfall.total_depth.mean() / config.n_years:.1f} mm")
        print(f"  Max total rainfall: {rainfall.total_depth.max():.1f} mm")
    
    if verbose:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
    
    return SimulationResults(
        config=config,
        elevation=elevation,
        stratigraphy=stratigraphy,
        wind_structures=wind_structures,
        orographic_low=orographic_low,
        storms=all_storms,
        rainfall=rainfall,
        climate_trends=climate_trends,
    )


def _plot_basic_results(results: SimulationResults) -> plt.Figure:
    """Create basic results plot."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Elevation
    im0 = axes[0, 0].imshow(results.elevation.T, origin='lower', cmap='terrain')
    axes[0, 0].set_title('Surface Elevation')
    plt.colorbar(im0, ax=axes[0, 0], label='m')
    
    # Wind structures composite
    composite = np.zeros((*results.elevation.shape, 3))
    composite[..., 0] = results.wind_structures.barrier_mask * 0.8
    composite[..., 1] = results.wind_structures.channel_mask * 0.8
    composite[..., 2] = (results.wind_structures.basin_labels > 0) * 0.8
    axes[0, 1].imshow(np.transpose(composite, (1, 0, 2)), origin='lower')
    axes[0, 1].set_title('Wind Structures (R=barriers, G=channels, B=basins)')
    
    # Orographic low pressure
    im2 = axes[1, 0].imshow(results.orographic_low.T, origin='lower', cmap='Blues')
    axes[1, 0].set_title('Orographic Low Pressure Likelihood')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Average annual rainfall
    n_years = results.config.n_years
    annual_rain = results.rainfall.total_depth / n_years
    im3 = axes[1, 1].imshow(annual_rain.T, origin='lower', cmap='Blues')
    axes[1, 1].set_title(f'Average Annual Rainfall ({n_years} years)')
    plt.colorbar(im3, ax=axes[1, 1], label='mm/year')
    
    for ax in axes.flat:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Quantum-Seeded Terrain and Weather Simulation")
    print("=" * 50)
    
    # Check for scipy (needed for some functions)
    try:
        from scipy.ndimage import binary_dilation, label
        from scipy.special import gamma
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        print("WARNING: scipy not available. Some features may be limited.")
    
    # Create configuration
    config = SimulationConfig(
        nx=128,
        ny=128,
        n_years=5,
        elevation_min=0.0,
        elevation_max=500.0,
    )
    
    # Run simulation
    print(f"\nRunning {config.n_years}-year simulation on {config.nx}x{config.ny} grid...")
    
    results = run_multi_year_weather_simulation(config, verbose=True)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Grid size: {config.nx} x {config.ny}")
    print(f"Simulation years: {config.n_years}")
    print(f"Total storms: {len(results.storms)}")
    print(f"Mean annual rainfall: {results.rainfall.total_depth.mean() / config.n_years:.1f} mm")
    print(f"Elevation range: {results.elevation.min():.1f} - {results.elevation.max():.1f} m")
    
    if results.stratigraphy:
        print(f"Stratigraphic layers: {len(results.stratigraphy.layer_order)}")
    
    # Plot if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")
        
        # Basic results
        fig1 = _plot_basic_results(results)
        fig1.savefig('/workspace/simulation_results.png', dpi=150, bbox_inches='tight')
        print("  Saved: simulation_results.png")
        
        # Wind structures debug
        fig2 = plot_wind_structures_debug(results.elevation, results.wind_structures)
        fig2.savefig('/workspace/wind_structures.png', dpi=150, bbox_inches='tight')
        print("  Saved: wind_structures.png")
        
        # Stratigraphic cross-sections
        if results.stratigraphy:
            fig3 = plot_cross_sections_xy(results.stratigraphy)
            fig3.savefig('/workspace/stratigraphy_cross_sections.png', dpi=150, bbox_inches='tight')
            print("  Saved: stratigraphy_cross_sections.png")
        
        plt.close('all')
        print("\nAll plots saved successfully!")
    else:
        print("\nMatplotlib not available - skipping plots")
    
    print("\nSimulation complete!")
