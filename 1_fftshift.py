import numpy as np
import cv2
from matplotlib import pyplot as plt

def fourier_transform(img_addr):
    # Read the image
    image = cv2.imread(img_addr, cv2.IMREAD_GRAYSCALE)
    # FFT shift
    fft_image = np.fft.fft2(image)

    # save
    cv2.imwrite('fft_result.tif', np.abs(fft_image))


# Call the function with the image address
img_addr = "../data/0-Image512/Glas_LCvolume001-7_org.tif"
fourier_transform(img_addr)

