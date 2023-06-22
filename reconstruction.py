import numpy as np

def reconstruct(kspace_data, autocalibration_data, acceleration_factor):
    autocalibration_size = autocalibration_data.shape[0]
    """
    for i in range(autocalibration_size):
        for j in range(autocalibration_size):
                print(autocalibration_data[i, j])
    """
    #half_size = autocalibration_size // 2
    half_size = 8 
    
    reconstructed_kspace = np.zeros(kspace_data.shape, dtype=np.complex128)
    for row in range(0, kspace_data.shape[0]):
        for col in range(0, kspace_data.shape[1]):
            if (row < half_size or row >= kspace_data.shape[0]-half_size or col < half_size or col >= kspace_data.shape[1]-half_size):
                reconstructed_kspace[row, col] = autocalibration_data[row, col] 
            else:
                if kspace_data[row, col] == 0:
                    reconstructed_kspace[row, col] = np.real(np.average(autocalibration_data[row-half_size:row+half_size, col-half_size:col+half_size])) + np.imag(autocalibration_data[row, col])
                    #reconstructed_kspace[row, col] = np.real(reconstructed_value).astype(np.uint) 
                    #print("recon: ", reconstructed_kspace[row, col])
                else:
                    reconstructed_kspace[row, col] = kspace_data[row, col] 
                    #print("origin: ", reconstructed_kspace[row, col])
    return reconstructed_kspace

