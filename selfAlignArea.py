import numpy as np
from PIL import Image

def generate_autocalibration_region(kspace, autocalibration_size):
    center_row = kspace.shape[0] // 2
    center_col = kspace.shape[1] // 2

    half_size = autocalibration_size // 2
    autocalibration_region = kspace[center_row-half_size:center_row+half_size,
                                    center_col-half_size:center_col+half_size]

    return autocalibration_region

def alignArea(kspace_image_path, out_file):
    kspace_image = Image.open(kspace_image_path)

    kspace_array = np.array(kspace_image)

    #autocalibration_size = kspace_array.shape[0] #512
    autocalibration_size = 512

    autocalibration_region = generate_autocalibration_region(kspace_array, autocalibration_size)
    """
    size = autocalibration_region.shape[0]
    for i in range(size):
        for j in range(size):
        print(autocalibration_region[i, j])
    """

    autocalibration_image = Image.fromarray(autocalibration_region)
    autocalibration_image.save(out_file)

if __name__ == "__main__":
	kspace_image_path = 'out_norm.tif'
	out_file = 'autocali_' + kspace_image_path
	alignArea(kspace_image_path, out_file)
