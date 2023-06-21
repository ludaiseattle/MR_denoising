import numpy as np
from PIL import Image

def inverse_fourier_transform(kspace):
    image = np.fft.ifftshift(kspace)  
    image = np.fft.ifft2(image)      
    image = np.abs(image)            
    image = np.uint8(image)         
    return image

def ifft(kspace_path, output_path):
    kspace_image = Image.open(kspace_path)
    kspace = np.array(kspace_image)

    reconstructed_image = inverse_fourier_transform(kspace)

    reconstructed_image_pil = Image.fromarray(reconstructed_image)
    reconstructed_image_pil.save(output_path)


if __name__ == "__main__":
    kspace_path = "./out_norm.tif"
    output_path = 'ifft_image.tif'
    ifft(kspace_path, output_path)
