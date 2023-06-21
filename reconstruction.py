import numpy as np
import cv2
from PIL import Image

#res1 = cv2.normalize(res1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
def grappa_reconstruction(kspace_data, autocalibration_data, acceleration_factor):
	autocalibration_size = autocalibration_data.shape[0]
	"""
	for i in range(autocalibration_size):
	    for j in range(autocalibration_size):
                print(autocalibration_data[i, j])
	"""
	#half_size = autocalibration_size // 2
	half_size = 2
    
	reconstructed_kspace = np.zeros(kspace_data.shape, dtype=np.uint8)
	for row in range(0, kspace_data.shape[0]):
		for col in range(0, kspace_data.shape[1]):
			if (row < half_size or row >= kspace_data.shape[0]-half_size or col < half_size or col >= kspace_data.shape[1]-half_size):
				reconstructed_kspace[row, col] = autocalibration_data[row, col] 
			else:
				if kspace_data[row, col] == 0:
					reconstructed_value = np.average(autocalibration_data[row-half_size:row+half_size, col-half_size:col+half_size])
					reconstructed_kspace[row, col] = np.real(reconstructed_value).astype(np.uint) 
					print("recon: ", reconstructed_kspace[row, col])
				else:
					reconstructed_kspace[row, col] = kspace_data[row, col] 
					print("origin: ", reconstructed_kspace[row, col])
	return reconstructed_kspace

def reconstruct(kspace_data_path, autocalibration_data_path, out_file):
	#read
	kspace_data = Image.open(kspace_data_path)
	kspace_data = np.array(kspace_data)

	autocalibration_data = Image.open(autocalibration_data_path)
	autocalibration_data = np.array(autocalibration_data)

	print("autocalibration_data shape:", autocalibration_data.shape)
	print("kspace_data shape:", kspace_data.shape)

	acceleration_factor = 2

	reconstructed_kspace = grappa_reconstruction(kspace_data, autocalibration_data, acceleration_factor)

	recons_image = Image.fromarray(np.real(reconstructed_kspace))
	recons_image.save(out_file)

if __name__ == "__main__":
	kspace_data_path = 'samp2_out_norm.tif'
	autocalibration_data_path = 'autocali_out_norm.tif'
	out_file = 'reconstructed_' + kspace_data_path
	reconstruct(kspace_data_path, autocalibration_data_path, out_file)
