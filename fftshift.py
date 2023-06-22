import numpy as np
from numpy import fft
import tifffile as tiff
from scipy import fftpack

def print_info(img, shifted_fft):
    print("--------------------")
    print("img shape: ", img.shape)
    print("img type: ", type(img))
    print("img value type: ", type(img[0][0]))
    print("img min value: ", img.min())
    print("img max value", img.max())
    print("--------------------")
    print("fft shape: ", shifted_fft.shape)
    print("fft type: ", type(shifted_fft))
    print("fft value type: ", type(shifted_fft[0][0]))
    print("fft min value: ", shifted_fft.min())
    print("fft max value: ", shifted_fft.max())
    print("--------------------")

def scipy_fftshift(img_addr):
    img = tiff.imread(img_addr)
    sci_fft = fftpack.fft(img)
    shifted_sci = fftpack.fftshift(sci_fft)
    #print_info(img, shifted_sci)
    return shifted_sci

def ifftshift(kspace):
    image = np.fft.ifftshift(kspace)  
    image = np.fft.ifft2(image)      
    amplitude_spectrum = np.abs(image)
    normalized_amplitude_spectrum = (amplitude_spectrum - np.min(amplitude_spectrum)) / (np.max(amplitude_spectrum) - np.min(amplitude_spectrum))
    scaled_amplitude_spectrum = normalized_amplitude_spectrum * 255
    scaled_amplitude_spectrum = scaled_amplitude_spectrum.astype(np.uint8)
    return scaled_amplitude_spectrum

def fftshift(img_addr):
    img = tiff.imread(img_addr)
    #img = img.astype(float)

    fft_tif = fft.fft2(img)
    shifted_fft = fft.fftshift(fft_tif)

    #amplitude_spectrum = np.abs(shifted_fft)
    #phase_spectrum = np.angle(shifted_fft)

    #print_info(img, shifted_fft)

    return shifted_fft

def normalize_amplitude_spectrum(amplitude_spectrum):
    log_amplitude = 20 * np.log10(1 + amplitude_spectrum)
    normalized_amplitude = (log_amplitude - np.min(log_amplitude)) / (np.max(log_amplitude) - np.min(log_amplitude))
    normalized_amplitude_scaled = normalized_amplitude * 255
    ret = normalized_amplitude_scaled.astype(np.uint8)
    return ret 

def get_amplitude_spectrum(img_addr, out_addr):
    shifted_fft = fftshift(img_addr)

    amplitude_spectrum = np.abs(shifted_fft)
    #print(amplitude_spectrum.shape)
    #print(type(amplitude_spectrum[0][0]))
    print(amplitude_spectrum.max())
    print(amplitude_spectrum.min())

    amplitude_spectrum = normalize_amplitude_spectrum(amplitude_spectrum)

    print(amplitude_spectrum.max())
    print(amplitude_spectrum.min())

    tiff.imwrite(out_addr, amplitude_spectrum)

def get_phase_spectrum(img_addr, out_addr):
    img = tiff.imread(img_addr)

    fft_tif = fft.fft(img)
    shifted_fft = fft.fftshift(fft_tif)
    phase_spectrum = np.angle(shifted_fft)
    #print(phase_spectrum.shape)
    #print(type(phase_spectrum[0][0]))
    tiff.imwrite(out_addr, phase_spectrum)

if __name__ == "__main__":
    img_addr = "../data/0-Image512/Glas_LCvolume001-7_org.tif"
    out_addr = "./fftshift_amplitude_spectrum.tif"
    phase_addr = "./fftshift_phase_spectrum.tif" 

    fftshift(img_addr)
    get_amplitude_spectrum(img_addr, out_addr)
    get_phase_spectrum(img_addr, phase_addr)
