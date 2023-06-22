import cv2
import numpy as np

def samp(fft):
    height, width = fft.shape[:2]

    image1 = np.zeros((height, width), dtype=np.complex128)
    image2 = np.zeros((height, width), dtype=np.complex128)

    for x in range(width):
        for y in range(height):
            if x % 2 == 0:
                image1[x, y] = fft[x, y]
            else:
                image2[x, y] = fft[x, y]

    return image1, image2
