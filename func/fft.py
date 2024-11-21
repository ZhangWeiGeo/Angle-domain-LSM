#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:25:20 2023

@author: zhangw0c
"""

import sys
import os
home_path = os.getenv("HOME"); pwd_path = os.getcwd();
func_dir  = os.path.join(home_path, "zw_lib", "python", "func")
sys.path.append(func_dir)
import plot_func as P

import numpy as np

parameters_arr = sys.argv[:];
if len(parameters_arr)==1:
    parameters_arr=[".py", "../test-in/ini_mig_arr-500-240.bin", "../test-in/in-noise.bin", "../test-in/noise.bin", 1, 1, 1, -20]

print("used for add random noise with snr");
print(parameters_arr);


in_file     = str(parameters_arr[1])#
ou_file1    = str(parameters_arr[2])#
ou_file2    = str(parameters_arr[3])#
nx          = int(parameters_arr[4])#
ny          = int(parameters_arr[5])#
nz          = int(parameters_arr[6])#