from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "fig3_43a.jpg"
OUTPUT = ROOT / "output"
CANDIDATES = OUTPUT / "candidates"
A_VALUES = (1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 2.2, 2.5, 2.7, 3.0)
FINAL_A = 2.7


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


def save_gray(image, path):
    image = np.floor(np.clip(image, 0, 255) + 0.5).astype(np.uint8)
    Image.fromarray(image).save(path)


def draw_comparison(images, values, filename, columns):
    rows = (len(values) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.4, rows * 4.3), layout="constrained")
    for ax, a in zip(np.asarray(axes).ravel(), values):
        ax.imshow(np.clip(images[a], 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"A = {a:.1f}")
        ax.axis("off")
    fig.savefig(OUTPUT / filename, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    CANDIDATES.mkdir(exist_ok=True)

    img = Image.open(INPUT).convert("L")
    gray = np.array(img, dtype=float)
    average = np.ones((3, 3), dtype=float) / 9
    blurred = convolve(gray, average)
    mask = gray - blurred

    save_gray(gray, OUTPUT / "orig.png")
    save_gray(blurred, OUTPUT / "blur.png")
    save_gray(scale_for_display(mask), OUTPUT / "mask_view.png")

    limit = np.abs(mask).max()
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    view = ax.imshow(mask, cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(view, ax=ax)
    ax.set_title("Signed Mask")
    ax.axis("off")
    fig.savefig(OUTPUT / "mask_signed.png", dpi=150, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(blurred, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("3x3 Mean Blur")
    signed = axes[2].imshow(mask, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title("Signed Mask")
    fig.colorbar(signed, ax=axes[2])
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT / "steps.png", dpi=150, facecolor="white")
    plt.close(fig)

    results = {}
    for a in A_VALUES:
        results[a] = a * gray - blurred
        save_gray(results[a], CANDIDATES / f"hb_a{a:.1f}.png")

    draw_comparison(results, A_VALUES[:6], "compare_main.png", 3)
    draw_comparison(results, A_VALUES[6:], "compare_extra.png", 2)

    final = results[FINAL_A]
    save_gray(final, OUTPUT / "final_a2.7.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(np.clip(final, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Final: A = 2.7")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT / "final_compare.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
