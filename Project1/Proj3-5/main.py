from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "fig3_40a.jpg"
OUT = ROOT / "output"
DATA = OUT / "data"
LAP_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)


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
    # 只遍历核，整幅图像使用数组切片计算。
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            result += kernel[row, col] * padded[row:row + height, col:col + width]
    return result


def display_scale(image: np.ndarray) -> np.ndarray:
    low, high = float(image.min()), float(image.max())
    return np.zeros_like(image) if low == high else (image - low) * 255.0 / (high - low)


def save_image(name: str, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image)).save(OUT / name)


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


def main() -> None:
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    with Image.open(INPUT) as source:
        original = np.asarray(source.convert("L"), dtype=np.float64)

    lap = spatial_filter(original, LAP_KERNEL)
    sharp = original + lap
    lap_view = display_scale(lap)

    for name, value in (("orig", original), ("lap", lap), ("lap_view", lap_view), ("sharp", sharp)):
        np.save(DATA / f"{name}.npy", value)
    save_image("orig.png", original)
    save_image("lap_view.png", lap_view)
    save_image("sharp.png", sharp)

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    show_image(ax, lap, "Signed Laplacian", signed=True)
    fig.savefig(OUT / "lap_signed.png", dpi=150, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    show_image(axes[0, 0], original, "Original")
    show_image(axes[0, 1], lap, "Signed Laplacian", signed=True)
    show_image(axes[1, 0], lap_view, "Laplacian view")
    show_image(axes[1, 1], sharp, "Sharpened")
    fig.savefig(OUT / "summary.png", dpi=150, facecolor="white")
    plt.close(fig)

    crop = (slice(150, 350), slice(170, 370))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    show_image(axes[0], original[crop], "Original detail")
    show_image(axes[1], sharp[crop], "Sharpened detail")
    fig.savefig(OUT / "detail.png", dpi=150, facecolor="white")
    plt.close(fig)

    stats = {
        "shape": list(original.shape),
        "lap_range": [float(lap.min()), float(lap.max())],
        "sharp_range": [float(sharp.min()), float(sharp.max())],
        "below_0_percent": float(100 * np.mean(sharp < 0)),
        "above_255_percent": float(100 * np.mean(sharp > 255)),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Proj3-5 output: {OUT}")


if __name__ == "__main__":
    main()

