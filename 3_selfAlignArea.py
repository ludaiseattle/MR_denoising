import numpy as np
from PIL import Image

def generate_autocalibration_region(kspace, autocalibration_size):
    center_row = kspace.shape[0] // 2
    center_col = kspace.shape[1] // 2

    half_size = autocalibration_size // 2
    autocalibration_region = kspace[center_row-half_size:center_row+half_size,
                                    center_col-half_size:center_col+half_size]

    return autocalibration_region

kspace_image_path = 'samp2_out_norm.tif'
kspace_image = Image.open(kspace_image_path)

kspace_array = np.array(kspace_image)

autocalibration_size = 512

autocalibration_region = generate_autocalibration_region(kspace_array, autocalibration_size)

autocalibration_image = Image.fromarray(autocalibration_region)
autocalibration_image.save('autocali_' + kspace_image_path)

