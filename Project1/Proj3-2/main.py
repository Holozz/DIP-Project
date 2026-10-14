from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "input" / "fig3_8a.jpg"
OUTPUT = ROOT / "output"


def histogram(gray):
    return np.bincount(gray.ravel(), minlength=256)


def equalize(gray):
    hist_orig = histogram(gray)
    prob_orig = hist_orig / gray.size
    cdf = np.cumsum(prob_orig)
    mapping = np.round(255 * cdf).astype(np.uint8)
    result = mapping[gray]
    prob_result = histogram(result) / result.size
    return prob_orig, cdf, mapping, result, prob_result


def draw_hist(values, title, filename):
    plt.figure(figsize=(7, 4))
    plt.bar(range(256), values, width=1)
    plt.xlim(0, 255)
    plt.xlabel("Gray Level")
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=150)
    plt.close()


def main():
    OUTPUT.mkdir(exist_ok=True)

    img = Image.open(INPUT).convert("L")
    gray = np.array(img)
    hist_orig, cdf, mapping, result, hist_result = equalize(gray)

    Image.fromarray(gray).save(OUTPUT / "orig.png")
    Image.fromarray(result).save(OUTPUT / "eq.png")
    draw_hist(hist_orig, "Original Histogram", "hist_orig.png")
    draw_hist(hist_result, "Equalized Histogram", "hist_eq.png")

    plt.figure(figsize=(6, 4))
    plt.plot(255 * cdf, label="255 * CDF")
    plt.step(range(256), mapping, where="mid", label="Rounded Mapping")
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.xlabel("Input Gray Level")
    plt.ylabel("Output Gray Level")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT / "map.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(result, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Equalized")
    axes[0, 2].plot(255 * cdf, label="255 * CDF")
    axes[0, 2].step(range(256), mapping, where="mid", label="Rounded Mapping")
    axes[0, 2].set(xlim=(0, 255), ylim=(0, 255), title="Mapping")
    axes[0, 2].legend()
    axes[1, 0].bar(range(256), hist_orig, width=1)
    axes[1, 0].set(xlim=(0, 255), title="Original Histogram", ylabel="Probability")
    axes[1, 1].bar(range(256), hist_result, width=1)
    axes[1, 1].set(xlim=(0, 255), title="Equalized Histogram", ylabel="Probability")
    axes[1, 2].axis("off")
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    fig.savefig(OUTPUT / "summary.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
