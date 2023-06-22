#import cupy as np
import numpy as np
#cp.cuda.Device().use()

def reconstruct(kspace_data, autocalibration_data, acceleration_factor):
    autocalibration_size = autocalibration_data.shape[0]
    """
    for i in range(autocalibration_size):
        for j in range(autocalibration_size):
                print(autocalibration_data[i, j])
    """
    #threshold = autocalibration_size // 2
    threshold = 8 

    top = 0
    bottom = 0
    left = 0
    right = 0
    size = kspace_data.shape[0]
    
    reconstructed_kspace = np.zeros(kspace_data.shape, dtype=np.complex128)
    for row in range(0, kspace_data.shape[0]):
        for col in range(0, kspace_data.shape[1]):
            if row < threshold:
                top = 0
            else:
                top = row - threshold 

            if row > size - threshold:
                bottom = size
            else:
                bottom = row + threshold

            if col < threshold:
                left = 0
            else:
                left = col - threshold

            if col > size - threshold:
                right = size
            else:
                right = col + threshold

            if kspace_data[row, col] == 0:
                reconstructed_kspace[row, col] = np.real(np.average(autocalibration_data[top:bottom, left:right])) + np.imag(autocalibration_data[row, col])
                #print("recon: ", reconstructed_kspace[row, col])
            else:
                reconstructed_kspace[row, col] = kspace_data[row, col] 
                #print("origin: ", reconstructed_kspace[row, col])
    return reconstructed_kspace

