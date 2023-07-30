import argparse
import random
import glob
import tifffile as tiff
import os
from fftshift import fftshift, ifftshift
from downsampling import star_sampling
from utils import save_amplitude

def print_info(shifted_fft):
    print("--------------------")
    print("fft shape: ", shifted_fft.shape)
    print("fft type: ", type(shifted_fft))
    print("fft value type: ", type(shifted_fft[0][0]))
    print("fft min value: ", shifted_fft.min())
    print("fft max value: ", shifted_fft.max())
    print("--------------------")

def get_outname(file, output_folder, suffix):
    out_file = os.path.basename(file)
    out_file, extension = os.path.splitext(out_file)
    out_file = os.path.join(output_folder, out_file)
    out1 = out_file + "_" + suffix + "1.tif"
    out2 = out_file + "_" + suffix + "2.tif"
    return out1, out2

def whole_flow(input_folder, output_folder):
    file_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    for file in file_paths:
        fft = fftshift(file)
        samp1, samp2 = star_sampling(fft, 8, 0, 4)
        #samp1, samp3, samp2, samp4 = horiz_samp_four(fft, 0.1)
        #samp1, samp2 = spiral_sampling(fft)
        
        #for test
        us1, us2 = get_outname(file, output_folder, "downsamp")
        save_amplitude(us1, samp1)
        save_amplitude(us2, samp2)
        ###
        out1, out2 = get_outname(file, output_folder, "sample")
        ifft1 = ifftshift(samp1)
        ifft2 = ifftshift(samp2)
        tiff.imwrite(out1, ifft1)
        tiff.imwrite(out2, ifft2)

def main(parser, args):
    flow_num = args.flow
    if flow_num == "whole":
        input_folder = args.input
        output_folder = args.output
        whole_flow(input_folder, output_folder)
    else:
        print("Invalid flow number.")
        parser.print_usage()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--flow", type=str, help="Flow steps(str): 1. whole")
    parser.add_argument("-i", "--input", default="", type=str, help="Input folder")
    parser.add_argument("-o", "--output", default="", type=str, help="Output folder")
    args = parser.parse_args()


    main(parser, args)

