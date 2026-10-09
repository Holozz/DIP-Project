from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("src//3.8(a).jpg").convert("L")
# 把图片灰度值转换成数组存储
gray = np.array(img)
def H(gray):
    # 统计灰度值出现的次数
    hist = np.bincount(gray.ravel(), minlength=256)
    # for gray_value, count in enumerate(hist):
    #     print(f"灰度值 {gray_value}: {count} 个像素")
    # print(hist)
    return hist

def T(r):
    s = r.copy()
    hist_s = np.zeros(256, dtype=int)
    s[0] = r[0]
    for k in range(1, 256):
        s[k] = s[k-1] + r[k]
    print(s)
    
    s = np.round(s * 255).astype(int)
    hist_s[int(s[0])] += hist_origin[0]
    for k in range(1, 256):
        hist_s[int(s[k])] += hist_origin[k]
    
    return s, hist_s

def draw_hist(hist, name):
    plt.bar(range(256), hist)
    plt.savefig(name+"_normalized_histogram.png")
    plt.xlabel("Gray Level")
    plt.ylabel("Normalized Number of Pixels")
    plt.show()
    plt.clf()

hist_origin = H(gray)
normalized_hist_origin = hist_origin / gray.size
draw_hist(normalized_hist_origin, "origin")

balanced_s, balanced_hist_origin = T(normalized_hist_origin)
balanced_hist_origin = balanced_hist_origin / gray.size
draw_hist(balanced_hist_origin, "enhanced")


# ⑤ 计算 CDF
cdf = np.cumsum(normalized_hist_origin)
# ⑥ 得到灰度映射表
print(balanced_s)
balanced_s = np.round(255 * cdf).astype(int)
print(balanced_s)
# ⑦ 用映射表处理原图
balanced_gray = balanced_s[gray]

plt.imshow(balanced_gray, cmap="gray")
plt.axis("off")
plt.savefig("src//enhanced.jpg")
plt.show()