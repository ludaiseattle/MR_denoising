import argparse
import glob
import os
from fftshift_norm import fftshift_norm
from undersampling import samp 

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

def process_flow3(input_folder, output_folder):
	pass

def main(parser, flow_num, input_folder, output_folder):
    if flow_num == "fftshift":
        fftshift(input_folder, output_folder)
    elif flow_num == "undersampling":
        undersampling_flow(input_folder, output_folder)
    elif flow_num == "3":
        process_flow3(input_folder, output_folder)
    else:
        print("Invalid flow number.")
        parser.print_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--flow", type=str, help="Flow steps(str): 1. fftshift 2. undersampling")
    parser.add_argument("-i", "--input", type=str, help="Input folder")
    parser.add_argument("-o", "--output", type=str, help="Output folder")
    args = parser.parse_args()

    flow_number = args.flow
    input_folder = args.input
    output_folder = args.output

    main(parser, flow_number, input_folder, output_folder)

