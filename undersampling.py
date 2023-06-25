import cv2
import numpy as np

def is_central(cthred, x, y, width, height):
    if ((x > width//2 - cthred and x < width//2 + cthred) 
    and (y > height//2 - cthred and y < height//2 + cthred)):
        return True
    else:
        return False

def samp(fft):
    height, width = fft.shape[:2]

    image1 = np.zeros((height, width), dtype=np.complex128)
    image2 = np.zeros((height, width), dtype=np.complex128)

    cthred = (width//2)*(8/10)

    for x in range(width):
        for y in range(height):
            if x % 2 == 0 or is_central(cthred, x, y, width, height):
            #if x % 2 == 0:
                image1[x, y] = fft[x, y]
            if x % 2 == 1 or is_central(cthred, x, y, width, height):
            #if x % 2 == 1:
                image2[x, y] = fft[x, y]

    return image1, image2
