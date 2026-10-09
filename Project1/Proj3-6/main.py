from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "fig3_43a.jpg"
OUT = ROOT / "output"
DATA = OUT / "data"
CANDIDATES = OUT / "candidates"
MAIN_A = (1.0, 1.2, 1.4, 1.5, 1.7, 2.0)
EXTRA_A = (2.2, 2.5, 2.7, 3.0)
FINAL_A = 2.7


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(image, 0, 255) + 0.5).astype(np.uint8)


def spatial_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    # if image.ndim != 2 or kernel.ndim != 2 or any(n % 2 == 0 for n in kernel.shape):
    #     raise ValueError("image must be 2-D and kernel dimensions must be odd")
    ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((ry, ry), (rx, rx)), mode="edge")
    kernel = kernel[::-1, ::-1]
    result = np.zeros_like(image, dtype=np.float64)
    height, width = image.shape
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            result += kernel[row, col] * padded[row:row + height, col:col + width]
    return result


def display_scale(image: np.ndarray) -> np.ndarray:
    low, high = float(image.min()), float(image.max())
    return np.zeros_like(image) if low == high else (image - low) * 255.0 / (high - low)


def save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image)).save(path)


def show_image(ax: Axes, image: np.ndarray, title: str, signed: bool = False) -> None:
    if signed:
        limit = max(float(np.abs(image).max()), 1.0)
        view = ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit)
        fig = ax.figure
        if fig is not None:
            fig.colorbar(view, ax=ax, fraction=0.045, pad=0.03)
    else:
        ax.imshow(to_uint8(image), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")


def save_compare(images: dict[float, np.ndarray], values: tuple[float, ...], name: str) -> None:
    cols = 3 if len(values) == 6 else 2
    rows = (len(values) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.3), layout="constrained")
    for ax, a in zip(np.asarray(axes).ravel(), values):
        show_image(ax, images[a], f"A = {a:.1f}")
    fig.savefig(OUT / name, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    for folder in (OUT, DATA, CANDIDATES):
        folder.mkdir(exist_ok=True)
    with Image.open(INPUT) as source:
        original = np.asarray(source.convert("L"), dtype=np.float64)

    blur = spatial_filter(original, np.ones((3, 3), dtype=np.float64) / 9.0)
    mask = original - blur
    mask_view = display_scale(mask)

    for name, value in (("orig", original), ("blur", blur), ("mask", mask)):
        np.save(DATA / f"{name}.npy", value)
    save_image(OUT / "orig.png", original)
    save_image(OUT / "blur.png", blur)
    save_image(OUT / "mask_view.png", mask_view)

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    show_image(ax, mask, "Signed mask", signed=True)
    fig.savefig(OUT / "mask_signed.png", dpi=150, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    show_image(axes[0], original, "Original")
    show_image(axes[1], blur, "3x3 mean blur")
    show_image(axes[2], mask, "Signed mask", signed=True)
    fig.savefig(OUT / "steps.png", dpi=150, facecolor="white")
    plt.close(fig)

    images: dict[float, np.ndarray] = {}
    stats: dict[str, dict[str, float]] = {}
    for a in MAIN_A + EXTRA_A:
        result = a * original - blur
        images[a] = result
        tag = f"hb_a{a:.1f}"
        np.save(CANDIDATES / f"{tag}.npy", result)
        save_image(CANDIDATES / f"{tag}.png", result)
        stats[f"{a:.1f}"] = {
            "raw_min": float(result.min()),
            "raw_max": float(result.max()),
            "saved_mean": float(to_uint8(result).mean()),
            "above_255_percent": float(100 * np.mean(result > 255)),
        }

    save_compare(images, MAIN_A, "compare_main.png")
    save_compare(images, EXTRA_A, "compare_extra.png")

    final = images[FINAL_A]
    np.save(DATA / f"final_a{FINAL_A:.1f}.npy", final)
    save_image(OUT / f"final_a{FINAL_A:.1f}.png", final)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    show_image(axes[0], original, "Original")
    show_image(axes[1], final, f"Final: A = {FINAL_A:.1f}")
    fig.savefig(OUT / "final_compare.png", dpi=150, facecolor="white")
    plt.close(fig)

    report = {
        "shape": list(original.shape),
        "final_a": FINAL_A,
        "mask_range": [float(mask.min()), float(mask.max())],
        "candidates": stats,
    }
    (OUT / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Proj3-6 output: {OUT}")


if __name__ == "__main__":
    main()

