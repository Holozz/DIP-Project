import matplotlib.pyplot as plt
from PIL import Image
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

H_low = gaussian_lowpass_filter(M, N, 10)
H_high = 1 - gaussian_lowpass_filter(M, N, 200)
H = H_low + H_high
g = freq_filter(f0, H)
Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).save("output/mixPass.png")

# 水平方向的频率索引，零频放在中央
x = np.arange(N) - N // 2

# 取穿过零频的中心行
plt.figure(figsize=(8, 4))
plt.plot(x, H_low[M // 2, :], label='Lowpass: D0=15')
plt.plot(x, H_high[M // 2, :], label='Highpass: D0=80')
plt.plot(x, H[M // 2, :], label='Mixed', linewidth=2)

plt.xlim(-150, 150)
plt.ylim(-0.05, 1.05)
plt.xlabel('Horizontal frequency bin')
plt.ylabel('Weight H')
plt.title('Filter frequency response: center row')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('output/filter_response.png', dpi=150)
plt.show()