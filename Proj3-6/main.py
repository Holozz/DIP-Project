# PROJECT 03-06
# # Unsharp Masking
# # (a) Use the programs developed in Projects 03-03 and 03-04 to implement high-
# # boost filtering, as given in Eq. (3.7-8). The averaging part of the process should be
# # done using the mask in Fig. 3.34(a).
# # (b) Download Fig. 3.43(a) and enhance it using the program you developed in (a).
# # Your objective is to choose constant A so that your result visually approximates Fig.3.43(d).
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("src//3.43(a).jpg").convert("L")
gray = np.array(img)
