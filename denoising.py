import argparse
import glob
import os
from fftshift_norm import fftshift_norm
from undersampling import samp 
from selfAlignArea import alignArea
from reconstruction import reconstruct

def fftshift(input_folder, output_folder):
    file_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    for file in file_paths:
        out_file = file.split("_org")[0]
        out_file = out_file + "_fft.tif"
        out_file = os.path.basename(out_file)
        out_file = os.path.join(output_folder, out_file)
        #print(file, out_file)
        fftshift_norm(file, out_file)

def undersampling_flow(input_folder, output_folder):
    file_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    for file in file_paths:
        out_file = os.path.basename(file)
        out_file, extension = os.path.splitext(out_file)
        out_file = os.path.join(output_folder, out_file)
        out1 = out_file + "_samp1.tif"
        out2 = out_file + "_samp2.tif"
        samp(file, out1, out2)

def alignArea_flow(input_folder, output_folder):
    file_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    for file in file_paths:
        out_file = os.path.basename(file)
        out_file, extension = os.path.splitext(out_file)
        out_file = out_file + "_align.tif"
        out_file = os.path.join(output_folder, out_file)
        #print(file, out_file)
        alignArea(file, out_file)
def reconstruction_flow(kspace_folder, autocali_folder, output_folder):
    #print(autocali_folder)
    file_paths = glob.glob(os.path.join(kspace_folder, "*.tif"))
    for file in file_paths:
        base_file = os.path.basename(file)
        align_file = base_file.split("_samp")[0]
        align_file = align_file + "_align.tif"
        align_file = os.path.join(autocali_folder, align_file)
        out_file, extension = os.path.splitext(base_file)
        out_file = out_file + "_reconstruct.tif"
        out_file = os.path.join(output_folder, out_file)
        reconstruct(file, align_file, out_file)

def main(parser, args):
    flow_num = args.flow
    if flow_num == "fftshift":
        input_folder = args.input
        output_folder = args.output
        fftshift(input_folder, output_folder)
    elif flow_num == "undersampling":
        input_folder = args.input
        output_folder = args.output
        undersampling_flow(input_folder, output_folder)
    elif flow_num == "alignArea":
        input_folder = args.input
        output_folder = args.output
        alignArea_flow(input_folder, output_folder)
    elif flow_num == "reconstruction":
        input_folder = args.input
        output_folder = args.output
        autocali_folder = args.autocali
        reconstruction_flow(input_folder, autocali_folder, output_folder)
    else:
        print("Invalid flow number.")
        parser.print_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--flow", type=str, help="Flow steps(str): 1. fftshift 2. undersampling 3. alignArea 4. reconstruction")
    parser.add_argument("-i", "--input", default="", type=str, help="Input folder")
    parser.add_argument("-o", "--output", default="", type=str, help="Output folder")
    parser.add_argument("-a", "--autocali", default="", type=str, help="autocalib folder")
    args = parser.parse_args()


    main(parser, args)

