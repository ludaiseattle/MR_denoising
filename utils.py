import numpy as np
import tifffile as tiff

def normalize_amplitude_spectrum(amplitude_spectrum):
    log_amplitude = 20 * np.log10(1 + amplitude_spectrum)
    normalized_amplitude = (log_amplitude - np.min(log_amplitude)) / (np.max(log_amplitude) - np.min(log_amplitude))
    normalized_amplitude_scaled = normalized_amplitude * 255
    ret = normalized_amplitude_scaled.astype(np.uint8)
    return ret 

def save_as_tif(address, kspace):
    tiff.imwrite(address, kspace)

def save_amplitude(address, kspace):
    save_as_tif(address, normalize_amplitude_spectrum(kspace))
