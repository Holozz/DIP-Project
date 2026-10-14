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
# H(u,v) = exp( -D²(u,v) / (2σ²) ),  D(u,v) = sqrt( (u - M/2)² + (v - N/2)² )
def gaussian_lowpass(M, N, sigma):
    u = np.arange(M)[:, None]
    v = np.arange(N)[None, :]
    D = (u - M/2)**2 + (v - N/2)**2
    exponent = -1 * D / (2.0 * sigma**2)
    return np.exp(exponent)

def freq_filter(f, H):
    F = np.fft.fftshift(np.fft.fft2(f))
    G = F * H
    return np.real(np.fft.ifft2(np.fft.ifftshift(G)))


img = Image.open("input/4.11(a).png").convert("L")
f0 = np.asarray(img)

F = np.fft.fft2(f0)
M, N = F.shape

H = gaussian_lowpass(M, N, 30)
g = freq_filter(f0, H)
Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).save("output/lowpass_sigma30.png")