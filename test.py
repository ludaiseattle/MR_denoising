import numpy as np
import cv2
from matplotlib import pyplot as plt

img_addr = "../data/0-Image512/Glas_LCvolume001-7_org.tif"
img = cv2.imread(img_addr, 0)

# Fourier transform
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dftshift = np.fft.fftshift(dft)
res1 = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))

# Normalize and convert to 8-bit unsigned integer
res1 = cv2.normalize(res1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Save the result
cv2.imwrite("./out.tif", res1)

# Perform inverse DFT
dftshift_reverse = np.fft.ifftshift(dftshift)
idft = cv2.idft(dftshift_reverse, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
idft = np.uint8(idft)

# Save the reversed image
cv2.imwrite("out_reverse.tif", idft)

