from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "fig3_40a.jpg"
OUTPUT = ROOT / "output"
LAPLACIAN = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float)


def convolve(gray, kernel):
    height, width = gray.shape
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(gray, ((py, py), (px, px)), mode="edge")
    result = np.zeros_like(gray, dtype=float)

    for i in range(height):
        for j in range(width):
            for row in range(kh):
                for col in range(kw):
                    result[i, j] += kernel[row, col] * padded[i + row, j + col]
    return result


def scale_for_display(image):
    low, high = image.min(), image.max()
    return (image - low) * 255 / (high - low)


def save_gray(image, filename):
    image = np.floor(np.clip(image, 0, 255) + 0.5).astype(np.uint8)
    Image.fromarray(image).save(OUTPUT / filename)


def main():
    OUTPUT.mkdir(exist_ok=True)

    img = Image.open(INPUT).convert("L")
    gray = np.array(img, dtype=float)
    laplacian = convolve(gray, LAPLACIAN)
    enhanced = gray + laplacian
    laplacian_view = scale_for_display(laplacian)

    save_gray(gray, "orig.png")
    save_gray(laplacian_view, "lap_view.png")
    save_gray(enhanced, "sharp.png")

    limit = np.abs(laplacian).max()
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    view = ax.imshow(laplacian, cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(view, ax=ax)
    ax.set_title("Signed Laplacian")
    ax.axis("off")
    fig.savefig(OUTPUT / "lap_signed.png", dpi=150, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    signed = axes[0, 1].imshow(laplacian, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[0, 1].set_title("Signed Laplacian")
    fig.colorbar(signed, ax=axes[0, 1])
    axes[1, 0].imshow(laplacian_view, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Laplacian View")
    axes[1, 1].imshow(np.clip(enhanced, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Sharpened")
    for ax in axes.ravel():
        ax.axis("off")
    fig.savefig(OUTPUT / "summary.png", dpi=150, facecolor="white")
    plt.close(fig)

    crop = (slice(150, 350), slice(170, 370))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    axes[0].imshow(gray[crop], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Detail")
    axes[1].imshow(np.clip(enhanced[crop], 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Sharpened Detail")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT / "detail.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
