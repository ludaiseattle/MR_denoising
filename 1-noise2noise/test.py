#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import torch.nn as nn

from datasets import load_dataset
from noise2noise import Noise2Noise

from argparse import ArgumentParser


def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-d', '--data', help='dataset root path', default='../data')
    parser.add_argument('-tt', '--test-target-dir', help='test target set path', default='/home/alyld7/data/1-whole_out/samp2')
    parser.add_argument('--load-ckpt', help='load model checkpoint')
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['gaussian', 'poisson', 'text', 'mc'], default='gaussian', type=str)
    parser.add_argument('-v', '--noise-param', help='noise parameter (e.g. sigma for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='image crop size', default=256, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)

    return parser.parse_args()

def get_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.tif'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

if __name__ == '__main__':
    """Tests Noise2Noise."""

    # Parse test parameters
    params = parse_args()
    test_list = get_files_in_folder(params.data)

    # Initialize model and test
    n2n = Noise2Noise(params, trainable=False)
    params.redux = False
    params.clean_targets = True
    n2n.load_model(params.load_ckpt)
    n2n.test(test_list, params.test_target_dir)
