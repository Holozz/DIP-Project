import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np

# PROJECT 04-03
# Lowpass Filtering
# (a) Implement the Gaussian lowpass filter in Eq. (4.3-7). You must be able to
# specify the size, M x N, of the resulting 2D function. In addition, you must be able
# to specify where the 2D location of the center of the Gaussian function.
# (b) Download Fig. 4.11(a) [this image is the same as Fig. 4.18(a)] and lowpass
# filter it to obtain Fig. 4.18(c)

# H(u,v) = exp( -D²(u,v) / (2*D0²) ),  D(u,v) = sqrt( (u - u0)² + (v - v0)² )

def gaussian_lowpass_filter(M, N, D0, center=None):
    if center is None:
        center = (M//2, N//2)
    u0, v0 = center
    # 构造坐标矩阵
    u = np.arange(M)[:, None]
    v = np.arange(N)[None, :]
    # 计算D^2
    D_2 = (u - u0)**2 + (v - v0)**2
    exponent = -1 * D_2 / (2.0 * D0**2)
    return np.exp(exponent)

def freq_filter(f, H):
    F = np.fft.fftshift(np.fft.fft2(f))
    G = F * H
    return np.real(np.fft.ifft2(np.fft.ifftshift(G)))


img = Image.open("input/4.11(a).png").convert("L")
f0 = np.asarray(img)
F = np.fft.fft2(f0)
M, N = F.shape

f = f0.astype(float)
dy, dx = np.gradient(f)
magnitude = np.sqrt(dx**2 + dy**2)
threshold = 10.0
w = 1 - np.exp(-(magnitude**2) / (2 * threshold**2))
# 将边缘保护范围向四周扩展约 7 像素
w_image = Image.fromarray(np.rint(w * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(size=15))
w = np.asarray(w_image, dtype=float) / 255.0

H1 = gaussian_lowpass_filter(M, N, D0=10)
H2 = gaussian_lowpass_filter(M, N, D0=40)

g_low = freq_filter(f, H2)
Image.fromarray(np.clip(g_low, 0, 255).astype(np.uint8)).save("output/lowPass.png")

g = freq_filter(f, H1)
g = (1 - w) * g + w * f
Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).save("output/weightLowPass.png")