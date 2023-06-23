import argparse
import glob
import tifffile as tiff
import os
from fftshift import fftshift, ifftshift
from undersampling import samp 
from reconstruction import reconstruct

def print_info(shifted_fft):
    print("--------------------")
    print("fft shape: ", shifted_fft.shape)
    print("fft type: ", type(shifted_fft))
    print("fft value type: ", type(shifted_fft[0][0]))
    print("fft min value: ", shifted_fft.min())
    print("fft max value: ", shifted_fft.max())
    print("--------------------")

def get_outname(file, output_folder):
    out_file = os.path.basename(file)
    out_file, extension = os.path.splitext(out_file)
    out_file = os.path.join(output_folder, out_file)
    out1 = out_file + "_samp1.tif"
    out2 = out_file + "_samp2.tif"
    return out1, out2

def whole_flow(input_folder, output_folder):
    file_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    for file in file_paths:
        fft = fftshift(file)
        samp1, samp2 = samp(fft)
        recons1 = reconstruct(samp1, fft, 2)
        recons2 = reconstruct(samp2, fft, 2)
        out1, out2 = get_outname(file, output_folder)
        ifft1 = ifftshift(recons1)
        ifft2 = ifftshift(recons2)
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
