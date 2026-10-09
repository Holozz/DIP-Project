# PROJECT 04-04
# Highpass Filtering Using a Lowpass Image
# (a) Subtract your image in Project 04-03(b) from the original to obtain a sharpened
# image, as in Eq. (4.4-14). You will note that the resulting image does not resemble
# the Gaussian highpass results in Fig. 4.26. Explain why this is so.
# (b) Adjust the variance of your Gaussian lowpass filter until the result obtained by
# image subtraction looks similar to Fig. 4.26(c). Explain your result

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
