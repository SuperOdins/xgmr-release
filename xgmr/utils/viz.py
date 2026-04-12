"""
시각화 유틸리티.

한국어: 매칭과 품질 맵을 쉽게 시각화한다.
English: Simple helpers to visualise matches and quality maps.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for headless environments
import matplotlib.pyplot as plt
from typing import Dict


def visualize_matches(rgb: np.ndarray, thermal: np.ndarray, matches: Dict) -> np.ndarray:
    """
    매칭 시각화 / Draw coarse matches between RGB and thermal.
    """

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(rgb)
    ax[0].set_title("RGB")
    ax[1].imshow(thermal, cmap="inferno")
    ax[1].set_title("Thermal")
    coords0 = matches["coords0"]
    coords1 = matches["coords1"]
    for c0, c1 in zip(coords0, coords1):
        ax[0].plot(c0[0], c0[1], "go", markersize=3)
        ax[1].plot(c1[0], c1[1], "go", markersize=3)
    for a in ax:
        a.axis("off")
    fig.canvas.draw()
    
    # Robust conversion using buffer_rgba (works with Agg backend)
    image = np.asarray(fig.canvas.buffer_rgba())
    image = image[:, :, :3] # RGBA -> RGB
    
    plt.close(fig)
    return image


def visualize_quality(Q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    품질 맵 시각화 / Visualise Q and gating weights.
    """

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(Q.squeeze(), cmap="viridis")
    ax[0].set_title("Q-map")
    ax[1].imshow(w.squeeze(), cmap="magma")
    ax[1].set_title("Weights")
    for a in ax:
        a.axis("off")
    fig.canvas.draw()
    
    # Robust conversion using buffer_rgba
    image = np.asarray(fig.canvas.buffer_rgba())
    image = image[:, :, :3] # RGBA -> RGB
    
    plt.close(fig)
    return image


def make_matching_figure(
    img0: np.ndarray,
    img1: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    color: np.ndarray = None,
    text: list[str] = [],
    dpi: int = 75,
    path: str = None,
) -> None:
    """
    Make variable-length matching figure for thesis with OpenCV for distinct styling.
    """
    H0, W0 = img0.shape[:2]
    H1, W1 = img1.shape[:2]
    
    # Handle multi-channel images
    if img0.ndim == 3 and img0.shape[2] > 3: img0 = img0[:, :, :3]
    if img1.ndim == 3 and img1.shape[2] > 3: img1 = img1[:, :, :3]
    
    # Layout: Side-by-side with padding
    pad = 20
    H, W = max(H0, H1) + 60, W0 + W1 + pad # Extra height for header
    
    out = 255 * np.ones((H, W, 3), dtype=np.uint8)
    
    # Place images (offset y by 60 for header)
    y_off = 60
    
    # Safe assignment with casting
    i0 = (img0 * 255.0).astype(np.uint8) if img0.max() <= 1.0 else img0.astype(np.uint8)
    i1 = (img1 * 255.0).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
    
    if i0.ndim == 2: i0 = cv2.cvtColor(i0, cv2.COLOR_GRAY2BGR)
    if i1.ndim == 2: i1 = cv2.cvtColor(i1, cv2.COLOR_GRAY2BGR)
        
    out[y_off:y_off+H0, :W0, :] = i0
    out[y_off:y_off+H1, W0+pad:, :] = i1
    
    # Draw Matches (Green Lines)
    # Using OpenCV lines for crispness
    shifted_pts1 = mkpts1 + np.array([W0 + pad, y_off])
    pts0_off = mkpts0 + np.array([0, y_off])
    
    # Draw lines
    for p0, p1 in zip(pts0_off, shifted_pts1):
        cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 1, cv2.LINE_AA)
        
    # Draw Points
    for p in pts0_off:
        cv2.circle(out, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
    for p in shifted_pts1:
        cv2.circle(out, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

    # Draw Text (Header)
    if text:
        # Join text with separator
        header_text = " | ".join(text)
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thick = 2
        (fw, fh), base = cv2.getTextSize(header_text, font, font_scale, thick)
        
        # Center text roughly or top-left
        tx = 20
        ty = 40
        
        # Black background for text
        cv2.putText(out, header_text, (tx, ty), font, font_scale, (0, 0, 0), thick + 2, cv2.LINE_AA) # Outline
        cv2.putText(out, header_text, (tx, ty), font, font_scale, (0, 0, 0), thick, cv2.LINE_AA)
    
    if path:
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    else:
        # Fallback if no path (headless check)
        pass

