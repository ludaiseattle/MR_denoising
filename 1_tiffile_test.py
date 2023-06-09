import numpy as np
import tifffile as tiff

def fourier_transform(img_addr):
    # Read the image using tifffile
    image = tiff.imread(img_addr)
    # Perform Fourier transform
    fft_image = np.fft.fft2(image)

    # Save as TIFF using tifffile
    tiff.imwrite('1_tifffile_test.tif', np.abs(fft_image))


# Call the function with the image address
img_addr = "../data/0-Image512/Glas_LCvolume001-7_org.tif"
fourier_transform(img_addr)

