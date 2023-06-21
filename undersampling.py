import cv2
import numpy as np

def samp(infile, out1, out2):
    mri_image = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)

    if mri_image is None:
        print("Failed to read the image.")
        exit()

    height, width = mri_image.shape[:2]

    image1 = np.zeros((height, width), dtype=np.uint8)
    image2 = np.zeros((height, width), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            if x % 2 == 0:
                image1[x, y] = mri_image[x, y]
            else:
                image2[x, y] = mri_image[x, y]

    cv2.imwrite(out1, image1)
    cv2.imwrite(out2, image2)

if __name__ == "__main__":
    name="out_norm.tif"
    out1 = "samp1_" + name
    out2 = "samp2_" + name
    samp(name, out1, out2)
