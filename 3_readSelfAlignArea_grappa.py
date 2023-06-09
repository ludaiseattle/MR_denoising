import numpy as np
from PIL import Image

def grappa_reconstruction(kspace_data, autocalibration_data, acceleration_factor):
	autocalibration_size = autocalibration_data.shape[0]
	#half_size = autocalibration_size // 2
	half_size = 2
    
	reconstructed_kspace = np.zeros(kspace_data.shape, dtype=np.complex64)
	for row in range(half_size, kspace_data.shape[0]-half_size):
		for col in range(half_size, kspace_data.shape[1]-half_size):
			if kspace_data[row, col] == 0:
				#reconstructed_value = np.sum(autocalibration_data * kspace_data[row-half_size:row+half_size, col-half_size:col+half_size]) / autocalibration_size**2
				reconstructed_value = np.average(autocalibration_data[row-half_size:row+half_size, col-half_size:col+half_size])
				reconstructed_kspace[row, col] = reconstructed_value
				print("recon: ", reconstructed_kspace[row, col])
			else:
				print("origin: ", np.sum(kspace_data[row, col]))
	return reconstructed_kspace

#read
kspace_data_path = 'samp2_out_norm.tif'
kspace_data = Image.open(kspace_data_path)
kspace_data = np.array(kspace_data)

autocalibration_data_path = 'autocali_samp2_out_norm.tif'
autocalibration_data = Image.open(autocalibration_data_path)
autocalibration_data = np.array(autocalibration_data)

print("autocalibration_data shape:", autocalibration_data.shape)
print("kspace_data shape:", kspace_data.shape)

acceleration_factor = 2

reconstructed_kspace = grappa_reconstruction(kspace_data, autocalibration_data, acceleration_factor)

recons_image = Image.fromarray(np.real(reconstructed_kspace))
recons_image.save('reconstructed_' + kspace_data_path)

