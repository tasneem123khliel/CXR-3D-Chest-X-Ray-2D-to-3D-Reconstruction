"""
Processing pipeline:
  1. Preprocess X-ray image
  2. Heuristic lung/heart segmentation (works without pre-trained weights)
  3. Physics-based depth estimation via Beer-Lambert approximation
  4. Smooth anatomical depth refinement
  5. 3-D surface/point-cloud generation
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, filters, morphology, measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_xray(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Normalize and enhance an X-ray for downstream processing.

    Pipeline:
      - Resize to square
      - Convert to float [0, 1]
      - CLAHE contrast enhancement
      - Gaussian denoising
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Normalize
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # CLAHE for better contrast
    image_uint8 = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_uint8).astype(np.float32) / 255.0

    # Mild Gaussian smooth
    enhanced = filters.gaussian(enhanced, sigma=0.8)

    return enhanced


# ──────────────────────────────────────────────────────────────────────────────
# 2. HEURISTIC SEGMENTATION  (no training required)
# ──────────────────────────────────────────────────────────────────────────────

def segment_chest(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Heuristic multi-region segmentation using classical CV.

    Returns masks for: left_lung, right_lung, heart, ribs, background
    """
    h, w = image.shape

    # ── Invert image (lungs appear dark = low X-ray absorption) ───────────────
    inv = 1.0 - image

    # ── Otsu threshold to find air-filled (lung) regions ─────────────────────
    thresh = filters.threshold_otsu(inv)
    binary = inv > (thresh * 0.85)

    # Clean up small noise
    binary = morphology.remove_small_objects(binary, min_size=500)
    binary = morphology.remove_small_holes(binary, area_threshold=2000)
    binary = ndimage.binary_fill_holes(binary)

    # ── Restrict lung search to central band (remove edges/arms) ──────────────
    mask = np.zeros_like(binary)
    mask[int(h * 0.05):int(h * 0.85), int(w * 0.05):int(w * 0.95)] = True
    binary = binary & mask

    # ── Keep two largest connected regions (left + right lung) ────────────────
    labeled, n_comp = ndimage.label(binary)
    if n_comp < 2:
        # Fallback: vertical split
        left_lung = np.zeros_like(binary)
        right_lung = np.zeros_like(binary)
        left_lung[:, :w // 2] = binary[:, :w // 2]
        right_lung[:, w // 2:] = binary[:, w // 2:]
    else:
        sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
        top2 = np.argsort(sizes)[-2:] + 1
        regions = [(labeled == i) for i in top2]
        # Sort by x-centroid: left vs right
        centroids = [np.argwhere(r).mean(axis=0)[1] for r in regions]
        if centroids[0] < centroids[1]:
            left_lung, right_lung = regions[0], regions[1]
        else:
            left_lung, right_lung = regions[1], regions[0]

    # ── Heart estimation: dense region between lungs, mid-lower chest ─────────
    heart_mask = np.zeros_like(binary)
    # High-density area between lungs
    combined_lung = left_lung | right_lung
    # Heart is roughly: center-left, bright on X-ray
    cx, cy = int(h * 0.35), int(w * 0.35)
    r_h, r_w = int(h * 0.15), int(w * 0.18)
    heart_candidates = np.zeros_like(binary)
    heart_candidates[cx:cx + r_h, cy:cy + r_w] = True
    # Keep only dense (bright = high attenuation = heart) regions
    dense = image > filters.threshold_otsu(image)
    heart_mask = heart_candidates & dense & ~combined_lung
    heart_mask = morphology.binary_closing(heart_mask, morphology.disk(5))

    # ── Rib estimation: curved high-intensity structures ──────────────────────
    rib_mask = (image > 0.65) & ~combined_lung & ~heart_mask
    rib_mask[:int(h * 0.1), :] = False   # remove top (shoulders)
    rib_mask[int(h * 0.8):, :] = False   # remove bottom

    # ── Spine: bright vertical stripe, center ─────────────────────────────────
    spine_band = int(w * 0.05)
    spine_mask = np.zeros_like(binary)
    spine_mask[:, w // 2 - spine_band:w // 2 + spine_band] = True
    spine_mask = spine_mask & (image > 0.6)

    background = ~(left_lung | right_lung | heart_mask | rib_mask)

    return {
        "left_lung":  left_lung.astype(np.uint8),
        "right_lung": right_lung.astype(np.uint8),
        "heart":      heart_mask.astype(np.uint8),
        "ribs":       rib_mask.astype(np.uint8),
        "spine":      spine_mask.astype(np.uint8),
        "background": background.astype(np.uint8),
    }


def masks_to_rgb(masks: dict[str, np.ndarray], image: np.ndarray) -> np.ndarray:
    """Render coloured overlay on grayscale X-ray."""
    rgb = np.stack([image, image, image], axis=-1)
    colors = {
        "left_lung":  [0.2, 0.7, 1.0],
        "right_lung": [0.2, 0.7, 1.0],
        "heart":      [1.0, 0.2, 0.2],
        "ribs":       [1.0, 0.85, 0.2],
        "spine":      [0.9, 0.5, 0.1],
    }
    for region, color in colors.items():
        if region in masks and masks[region].any():
            for c, val in enumerate(color):
                rgb[:, :, c] = np.where(masks[region], 0.4 * val + 0.6 * rgb[:, :, c], rgb[:, :, c])
    return np.clip(rgb, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. DEPTH ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────

def beer_lambert_depth(image: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Physics-based depth from X-ray intensity.

    X-ray follows Beer-Lambert: I = I₀·exp(−μd)
    → d ∝ −ln(I/I₀) ≈ −ln(I)    [assuming I₀ = 1 after normalisation]

    Dense tissue (bone, heart) → bright pixel → high attenuation → greater depth.
    Air (lungs) → dark pixel → less attenuation → lower depth.
    """
    depth = -np.log(image + epsilon)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + epsilon)
    return depth.astype(np.float32)


def anatomical_depth_refinement(
    raw_depth: np.ndarray,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Refine raw depth using anatomical priors.

    Approximate anterior-posterior anatomy (front of chest to back):
      skin/anterior ribs (0.1) → anterior lungs (0.2–0.4)
      heart (0.5) → posterior lungs (0.4–0.7)
      spine/posterior ribs (0.8–1.0)
    """
    depth = raw_depth.copy()
    h, w = depth.shape

    # ── Lung depth: gradient from anterior surface to posterior ───────────────
    lung_combined = masks.get("left_lung", np.zeros_like(depth)) | masks.get("right_lung", np.zeros_like(depth))
    if lung_combined.any():
        # Distance from lung edge as proxy for depth within lung
        dist_transform = ndimage.distance_transform_edt(lung_combined)
        dist_norm = dist_transform / (dist_transform.max() + 1e-8)
        depth = np.where(lung_combined, 0.15 + 0.50 * dist_norm, depth)

    # ── Heart: mid-depth, slightly behind anterior chest wall ─────────────────
    if masks.get("heart", np.zeros_like(depth)).any():
        heart_mask = masks["heart"].astype(bool)
        depth[heart_mask] = 0.55 + 0.10 * depth[heart_mask]

    # ── Ribs: wrap around lung surface ────────────────────────────────────────
    if masks.get("ribs", np.zeros_like(depth)).any():
        rib_mask = masks["ribs"].astype(bool)
        # Anterior ribs closer to camera (lower depth)
        y_coords = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
        depth[rib_mask] = 0.10 + 0.30 * y_coords[rib_mask]

    # ── Spine: deepest structure ───────────────────────────────────────────────
    if masks.get("spine", np.zeros_like(depth)).any():
        depth[masks["spine"].astype(bool)] = 0.85

    # ── Smooth to remove sharp discontinuities ────────────────────────────────
    depth = filters.gaussian(depth, sigma=4)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 3-D SURFACE DATA
# ──────────────────────────────────────────────────────────────────────────────

def build_3d_surface(
    image: np.ndarray,
    depth: np.ndarray,
    scale_z: float = 80.0,
    downsample: int = 2,
) -> dict:
    """
    Build X, Y, Z arrays and surface-colour array for a Plotly Surface plot.

    depth is normalised [0, 1]; Z = depth * scale_z
    """
    step = max(1, downsample)
    img_ds = image[::step, ::step]
    dep_ds = depth[::step, ::step]

    h, w = img_ds.shape
    X = np.linspace(0, 1, w)
    Y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(X, Y)
    Z = dep_ds * scale_z

    # Surface colour = grayscale X-ray intensity
    surf_color = img_ds

    return {"X": X, "Y": Y, "Z": Z, "color": surf_color}


def build_point_cloud(
    image: np.ndarray,
    depth: np.ndarray,
    masks: dict[str, np.ndarray],
    n_points: int = 6000,
) -> dict:
    """
    Sample coloured 3-D point cloud from depth map, colour-coded by anatomy.
    """
    region_colors = {
        "left_lung":  "#3DB8FF",
        "right_lung": "#3DB8FF",
        "heart":      "#FF4444",
        "ribs":       "#FFD700",
        "spine":      "#FF8C00",
    }

    all_x, all_y, all_z, all_colors, all_labels = [], [], [], [], []
    h, w = image.shape

    for region, hex_color in region_colors.items():
        mask = masks.get(region, np.zeros((h, w), dtype=np.uint8))
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        n_sample = min(n_points // len(region_colors), len(ys))
        idx = np.random.choice(len(ys), n_sample, replace=False)
        ys, xs = ys[idx], xs[idx]
        zs = depth[ys, xs] * 100

        all_x.extend(xs / w)
        all_y.extend(ys / h)
        all_z.extend(zs)
        all_colors.extend([hex_color] * n_sample)
        all_labels.extend([region.replace("_", " ").title()] * n_sample)

    return {
        "x": np.array(all_x),
        "y": np.array(all_y),
        "z": np.array(all_z),
        "colors": all_colors,
        "labels": all_labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    raw_image: np.ndarray,
    target_size: int = 512,
    depth_z_scale: float = 80.0,
    downsample: int = 2,
) -> dict:
    """
    End-to-end pipeline from raw image to 3-D data.

    Returns a dict with every intermediate result for display.
    """
    preprocessed = preprocess_xray(raw_image, target_size)
    masks = segment_chest(preprocessed)
    raw_depth = beer_lambert_depth(preprocessed)
    refined_depth = anatomical_depth_refinement(raw_depth, masks)
    overlay = masks_to_rgb(masks, preprocessed)
    surface_data = build_3d_surface(preprocessed, refined_depth, depth_z_scale, downsample)
    point_cloud = build_point_cloud(preprocessed, refined_depth, masks)

    # ── Depth colourmap (viridis-style) ───────────────────────────────────────
    depth_vis = (refined_depth * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    return {
        "preprocessed":   preprocessed,
        "masks":          masks,
        "overlay":        overlay,
        "raw_depth":      raw_depth,
        "refined_depth":  refined_depth,
        "depth_colored":  depth_colored.astype(np.float32) / 255.0,
        "surface":        surface_data,
        "point_cloud":    point_cloud,
    }
