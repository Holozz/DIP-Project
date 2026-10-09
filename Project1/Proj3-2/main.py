from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "fig3_8a.jpg"
OUT = ROOT / "output"
DATA = OUT / "data"


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(image, 0, 255) + 0.5).astype(np.uint8)


def histogram(image: np.ndarray) -> np.ndarray:
    return np.bincount(image.astype(np.int64).ravel(), minlength=256)


def equalize(image: np.ndarray) -> dict[str, np.ndarray]:
    hist_orig = histogram(image)
    prob = hist_orig.astype(np.float64) / image.size
    counts = np.cumsum(hist_orig, dtype=np.int64)
    cdf = counts.astype(np.float64) / image.size
    map_float = 255.0 * cdf
    # 用整数式完成 half-up 取整，避免浮点边界误差
    mapping = ((510 * counts + image.size) // (2 * image.size)).astype(np.uint8)
    result = mapping[image.astype(np.int64)]
    return {
        "hist_orig": hist_orig,
        "prob": prob,
        "cdf": cdf,
        "map_float": map_float,
        "map": mapping,
        "eq": result,
        "hist_eq": histogram(result),
    }


def save_image(name: str, image: np.ndarray) -> None:
    Image.fromarray(to_uint8(image)).save(OUT / name)


def save_hist(name: str, values: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    ax.bar(np.arange(256), values, width=1.0, color="#355f8d")
    ax.set(xlim=(-1, 256), xlabel="Gray level", ylabel="Pixel count", title=title)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig(OUT / name, dpi=150, facecolor="white")
    plt.close(fig)


def save_summary(image: np.ndarray, values: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(values["eq"], cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Equalized")
    axes[0, 2].plot(values["map_float"], label="255 * CDF")
    axes[0, 2].step(np.arange(256), values["map"], where="mid", alpha=0.65, label="Rounded")
    axes[0, 2].set(xlim=(0, 255), ylim=(0, 255), xlabel="Input gray level", ylabel="Output gray level", title="Mapping")
    axes[0, 2].legend()
    for ax, key, title in (
        (axes[1, 0], "hist_orig", "Original histogram"),
        (axes[1, 1], "hist_eq", "Equalized histogram"),
    ):
        ax.bar(np.arange(256), values[key], width=1.0, color="#355f8d")
        ax.set(xlim=(-1, 256), xlabel="Gray level", ylabel="Pixel count", title=title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1, 2].axis("off")
    for ax in axes[0, :2]:
        ax.axis("off")
    fig.savefig(OUT / "summary.png", dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    with Image.open(INPUT) as source:
        image = np.asarray(source.convert("L"), dtype=np.float64)

    values = equalize(image)
    np.save(DATA / "orig.npy", image)
    for name, value in values.items():
        np.save(DATA / f"{name}.npy", value)

    save_image("orig.png", image)
    save_image("eq.png", values["eq"])
    save_hist("hist_orig.png", values["hist_orig"], "Original histogram")
    save_hist("hist_eq.png", values["hist_eq"], "Equalized histogram")

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(values["map_float"], label="255 * CDF")
    ax.step(np.arange(256), values["map"], where="mid", alpha=0.65, label="Rounded")
    ax.set(xlim=(0, 255), ylim=(0, 255), xlabel="Input gray level", ylabel="Output gray level")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(OUT / "map.png", dpi=150, facecolor="white")
    plt.close(fig)
    save_summary(image, values)

    stats = {
        "shape": list(image.shape),
        "orig_mean": float(image.mean()),
        "eq_mean": float(values["eq"].mean()),
        "output_levels": int(np.count_nonzero(values["hist_eq"])),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Proj3-2 output: {OUT}")


if __name__ == "__main__":
    main()

