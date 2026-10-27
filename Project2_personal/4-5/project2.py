# PROJECT 04-05
# 基于频域相关定理实现模板匹配
# 原理：空间域互相关 ↔ 频域 F(u,v) * T*(u,v)

import os
from PIL import Image
import numpy as np

os.makedirs("output", exist_ok=True)


def freq_correlation(f, t):
    # 计算线性相关补零尺寸，避免周期混叠
    m, n = f.shape
    h, w = t.shape
    P = m + h - 1
    Q = n + w - 1

    # 零延拓到相同尺寸
    f_pad = np.zeros((P, Q))
    t_pad = np.zeros((P, Q))
    f_pad[:m, :n] = f
    t_pad[:h, :w] = t

    # 频域相关：FFT + 共轭相乘 + 逆变换
    F = np.fft.fft2(f_pad)
    T = np.fft.fft2(t_pad)
    corr = np.real(np.fft.ifft2(F * np.conj(T)))
    return corr, f_pad, t_pad


# 读取原图和模板，转灰度浮点数组
src_img = Image.open("inputs/4.41a.jpg").convert("L")
tpl_img = Image.open("inputs/4.41b.png").convert("L")

src = np.asarray(src_img, dtype=float)
tpl = np.asarray(tpl_img, dtype=float)
m, n = src.shape
h, w = tpl.shape

# 计算频域互相关
corr, f_pad, t_pad = freq_correlation(src, tpl)

# 在有效区域找最大值，即匹配位置
valid = corr[: m - h + 1, : n - w + 1]
y, x = np.unravel_index(np.argmax(valid), valid.shape)

# ---------- 保存所有单张素材图 ----------
# 1. 原图
Image.fromarray(np.uint8(src)).save("output/original.png")
# 2. 模板
Image.fromarray(np.uint8(tpl)).save("output/template.png")
# 3. 补零后原图
Image.fromarray(np.uint8(f_pad[:m, :n])).save("output/image_padded.png")
# 4. 补零后模板
Image.fromarray(np.uint8(t_pad)).save("output/template_padded.png")
# 5. 相关函数图（线性拉伸到0-255）
corr_show = 255 * (corr[:m, :n] - corr.min()) / (corr.max() - corr.min())
Image.fromarray(np.uint8(corr_show)).save("output/correlation.png")
# 6. 匹配区域裁剪图
match_patch = src[y:y + h, x:x + w]
Image.fromarray(np.uint8(match_patch)).save("output/matched_patch.png")

print(f"匹配完成，模板左上角坐标：x = {x}, y = {y}")
