from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
D0_VALUES = (5, 15, 30, 80, 230)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})


def read_image(name):
    return np.array(Image.open(ROOT / 'inputs' / name).convert('L'), dtype=float)


def uint8(image):
    return (np.clip(image, 0, 255) + 0.5).astype(np.uint8)


def scale(image):
    low, high = image.min(), image.max()
    return 255 * (image - low) / (high - low) if high > low else np.zeros_like(image)


def gaussian_lowpass(shape, d0, center=None):
    cy, cx = center if center is not None else (shape[0] // 2, shape[1] // 2)
    y, x = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * d0**2))


def filter_spectrum(spectrum, transfer):
    return np.fft.ifft2(np.fft.ifftshift(spectrum * transfer))


def save_image(out, name, image):
    Image.fromarray(uint8(image)).save(out / name)


def show(ax, image, title, signed=False):
    if signed:
        im = ax.imshow(image, cmap='RdBu_r', vmin=-255, vmax=255)
        ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.03)
    else:
        ax.imshow(uint8(image), cmap='gray', vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis('off')


def save_figure(fig, out, name):
    fig.savefig(out / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def lowpass_experiment(image, out):
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    lowpasses, transfers = {}, {}
    save_image(out, 'project_03_original.png', image)
    save_image(out, 'project_03_spectrum.png', scale(np.log1p(abs(spectrum))))
    for d0 in D0_VALUES:
        h = gaussian_lowpass(image.shape, d0)
        result = filter_spectrum(spectrum, h)
        lowpasses[d0], transfers[d0] = result.real, h
        save_image(out, f'project_03_lowpass_D0_{d0}.png', result.real)
    filtered_spectrum = spectrum * transfers[15]
    save_image(out, 'project_03_filtered_spectrum.png', scale(np.log1p(abs(filtered_spectrum))))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    panels = [(image, 'Original'), (scale(np.log1p(abs(spectrum))), 'Log spectrum'),
              (255 * transfers[15], 'H: D0 = 15'), (lowpasses[15], 'Lowpass: D0 = 15'),
              (scale(np.log1p(abs(filtered_spectrum))), 'Filtered spectrum')]
    for ax, (a, title) in zip(axes.flat, panels):
        show(ax, a, title)
    x = np.arange(image.shape[1]) - image.shape[1] // 2
    for d0 in (15, 30, 80):
        axes[1, 2].plot(x, transfers[d0][image.shape[0] // 2], label=f'D0={d0}')
    axes[1, 2].set(xlim=(-150, 150), title='H: center row', xlabel='Frequency', ylabel='H')
    axes[1, 2].legend()
    save_figure(fig, out, 'project_03_overview.png')
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    show(axes.flat[0], image, 'Original')
    for ax, d0 in zip(axes.flat[1:], D0_VALUES):
        show(ax, lowpasses[d0], f'D0 = {d0}')
    save_figure(fig, out, 'project_03_parameters.png')
    return lowpasses


def highpass_experiment(image, lowpasses, out):
    results = {}
    for d0 in (15, 30, 80):
        raw = image - lowpasses[d0]  # 保留浮点负值，保存PNG时才裁剪。
        results[d0] = raw
        save_image(out, f'project_04_highpass_D0_{d0}.png', raw)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    for row, d0 in enumerate((15, 80)):
        show(axes[row, 0], lowpasses[d0], f'Lowpass: D0 = {d0}')
        show(axes[row, 1], results[d0], f'Highpass: D0 = {d0}', signed=True)
        show(axes[row, 2], results[d0], f'Clipped: D0 = {d0}')
    save_figure(fig, out, 'project_04_overview.png')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, d0 in zip(axes, (15, 30, 80)):
        show(ax, results[d0], f'D0 = {d0}')
    save_figure(fig, out, 'project_04_parameters.png')


def boundary_experiment(image, original_lowpass, out):
    m, n = image.shape
    padded = np.pad(image, ((0, m), (0, n)))
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    # FFT尺寸翻倍，D0也翻倍，保持相同的空间模糊尺度。
    result = filter_spectrum(spectrum, gaussian_lowpass(padded.shape, 30)).real[:m, :n]
    difference = result - original_lowpass
    save_image(out, 'boundary_zero_padding.png', result)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    show(axes[0], original_lowpass, 'Periodic: D0 = 15')
    show(axes[1], result, 'Zero padding: D0 = 30')
    show(axes[2], difference, 'Difference', signed=True)
    save_figure(fig, out, 'boundary_comparison.png')


def correlation_experiment(image, template, out):
    m, n = image.shape
    h, w = template.shape
    shape = (m + h - 1, n + w - 1)
    fpad, tpad = np.zeros(shape), np.zeros(shape)
    fpad[:m, :n], tpad[:h, :w] = image, template
    fs, ts = np.fft.fft2(fpad), np.fft.fft2(tpad)
    product = fs * ts.conj()
    inverse = np.fft.ifft2(product)
    corr = inverse.real
    valid = corr[:m - h + 1, :n - w + 1]
    y, x = map(int, np.unravel_index(valid.argmax(), valid.shape))  # 模板左上角。
    for name, a in [('original', image), ('template', template), ('image_padded', fpad),
                    ('template_padded', tpad), ('correlation', scale(corr[:m, :n])),
                    ('match_patch', image[y:y + h, x:x + w])]:
        save_image(out, f'project_05_{name}.png', a)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    panels = [(image, 'Original'), (template, 'Template'), (tpad, 'Padded template'),
              (scale(corr[:m, :n]), 'Correlation'), (image, f'Match: ({x}, {y})'),
              (image[y:y + h, x:x + w], 'Matched patch')]
    for ax, (a, title) in zip(axes.flat, panels):
        show(ax, a, title)
    axes[1, 0].plot(x, y, 'r+', markersize=9)
    axes[1, 1].add_patch(Rectangle((x - .5, y - .5), w, h, fill=False, edgecolor='red', linewidth=1.5))
    save_figure(fig, out, 'project_05_overview.png')
    return x, y


def main():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    image = read_image('Fig0418(a)(characters_test_pattern).jpeg')
    letters, template = read_image('4.41a.jpg'), read_image('4.41b.png')
    lowpasses = lowpass_experiment(image, out)
    highpass_experiment(image, lowpasses, out)
    boundary_experiment(image, lowpasses[15], out)
    x, y = correlation_experiment(letters, template, out)
    print('模板匹配位置：', (x, y))


if __name__ == '__main__':
    main()
