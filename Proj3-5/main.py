#PROJECT 03-03 [Multiple Uses]
# Arithmetic Operations
# Write a computer program capable of performing the four arithmetic operations between
# two images. This project is generic, in the sense that it will be used in other projects to
# follow. (See comments on pages 112 and 116 regarding scaling). In addition to
# multiplying two images, your multiplication function must be able to handle multiplication
# of an image by a constant.
#PROJECT 03-04 [Multiple Uses]
# Spatial Filtering
# Write program to perform spatial filtering of an image (see Section 3.5 regarding
# implementation). You can fix the size of the spatial mask at 3 x 3, but the coefficients
# need to be variables that can be input into your program. This project is generic, in the
# sense that it will be used in other projects to follow.
#PROJECT 03-05
# Enhancement Using the Laplacian
# (a) Use the programs developed in Projects 03-03 and 03-04 to implement the
# Laplacian enhancement technique described in connection with Eq. (3.7-5). Use the
# mask shown in Fig. 3.39(d).
# (b) Duplicate the results in Fig. 3.40. You will need to download Fig. 3.40(a).

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def convovle(img, kernel:list):
    # 实现卷积操作
    res = np.zeros_like(img)
    L, H = img.shape[0], img.shape[1]
    l, h = len(kernel), len(kernel[0])

    padded_img = np.zeros((L+2*(l//2), H+2*(h//2)), dtype=int)
    padded_img[l//2:l//2+L, h//2:h//2+H] = img
    # print(padded_img)
    for i in range(L):
        for j in range(H):
            val = np.sum(kernel * padded_img[i:i+l, j:j+h])
            if val < 0:
                res[i, j] = 0
            elif val > 255:
                res[i, j] = 255
            res[i, j] = val
    return res

def laplacian_enhancement(img, mask = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]):
    # 实现拉普拉斯增强
    return img + convovle(img, mask)

img = Image.open("src//3.40(a).jpg").convert("L")

gray = np.array(img)

laplacian_enhanced = laplacian_enhancement(gray)

plt.imshow(laplacian_enhanced, cmap="gray")
plt.axis("off")
plt.savefig("laplacian_enhanced.jpg")
plt.show()


