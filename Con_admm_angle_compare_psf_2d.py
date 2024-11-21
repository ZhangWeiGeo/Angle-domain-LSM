#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 09:27:06 2023

@author: zhangw0c
"""

from Library import *
from Para import *

def imshow_and_write_km(input_arr, eps_name, data_name, full_model=False, velocity=False, density=False, color="gray", plot_min=0, plot_max=0, km_bool=True, title=""):
    
    [nx, nz] = input_arr.shape ;
    if km_bool:
        label_scale=1000.0 ;
        xlabel="Distance (km)" ;
        ylabel="Depth (km)" ;
    else:
        label_scale=1.0;
        xlabel="Distance (m)" ;
        ylabel="Depth (m)" ;
         
    if full_model:
        x1beg = 0 ;
        x1end = nx*dx/label_scale ;
            
        z1beg = 0 ;
        z1end = nz*dz/label_scale ;
    else:
        x1beg =  x_beg*dx         /label_scale ;
        x1end = (x_beg*dx + nx*dx)/label_scale ;
            
        z1beg =  z_beg*dz         /label_scale ;
        z1end = (z_beg*dz + nz*dz)/label_scale ;      
        
    if velocity:
        if km_bool:
            colorbar_label="Velocity (km/s)" ;
            input_arr = input_arr/1000.0 ;
        else:
            colorbar_label="Velocity (m/s)" ;
    elif density:
        if km_bool:
            colorbar_label="Velocity (kg/m3)" ;
            input_arr = input_arr/1000.0 ;
        else:
            colorbar_label="Velocity (g/cm3)" ;
            
    else:
        colorbar_label="Relative amplitude" ;
    
    if plot_min==0 and plot_max==0:
        vmin = input_arr.min() ;
        vmax = input_arr.max() ;
    else:
        vmin = plot_min ;
        vmax = plot_max ;
    
    P.imshow1(input_arr.T,  x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=(6, 3.5), output_name=eps_name + fig_type, eps_dpi=eps_dpi, title=title, xtop=True, pltshow=False);
    
    P.fwrite_file(data_name, input_arr);



    
psf1_name="/media/zhangjiwei/z16/paper/DE-AD-IDLSM/Graben/graben-vp-820-315-3-inv/admm-angle-vel-new/vel_cc_none_part2/output4/5e-4-0.2-0.2-0-0-0-0-250-5e-4-0.6-1/data/ini/0-psf-90-855-495.bin"
psf2_name="/media/zhangjiwei/z16/paper/DE-AD-IDLSM/Graben/graben-vp-820-315-3-inv/admm-angle-vel-new/vel_ex_0_part2/output4/5e-4-0.2-0.2-0-0-0-0-250-5e-4-0.6-1/data/ini/0-psf-90-855-495.bin"

psf_nx=855
psf_nz=495
psf_angle=90

shape_list=(psf_angle, psf_nx, psf_nz)
psf1_arr=np.zeros(shape_list, dtype=np.float32)
psf2_arr=np.zeros(shape_list, dtype=np.float32)

P.fread_file(psf1_name, psf1_arr, shape_list=shape_list);
P.fread_file(psf2_name, psf2_arr, shape_list=shape_list);


output = 1.0 * psf2_arr

#graben
sort_angle_list  = [15, 25, 35, 45, 55, 65, 75]
clip1_scale_list = [-0.12*10e-1, -0.12*10e-1, -0.12*10e-1, -0.12*10e-1, -0.12*10e-1, -0.12*10e-1, -0.12*10e-1, -0.12*10e-1, -0.12*10e-1, -0.12*10e-1]
clip2_scale_list = [+0.12*10e-1, +0.12*10e-1, +0.12*10e-1, +0.12*10e-1, +0.12*10e-1, +0.12*10e-1, +0.12*10e-1, +0.12*10e-1, +0.12*10e-1, +0.12*10e-1]
colorbar_ticks=((), (), (), (), (), (), (), (), (), (), (), (), (), (), ())


#stremear
# sort_angle_list  = [0, 5, 10, 15, 20, 25, 30, 35, 40]
# clip1_scale_list = [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6, -0.6]
# clip2_scale_list = [+0.6, +0.6, +0.6, +0.6, +0.6, +0.6, +0.6, +0.6, +0.6]
# colorbar_ticks=[ (), (), (), (), (), (), (), (), (), (), (), (), (), ()]

figsize_list=( (6, 3.5), )



scalex=show_wx/wx
scalez=show_wz/wz

xlabel="Distance (km)" ;
ylabel="Depth (km)" ;
label_scale = 1000.0

x1beg =  x_beg*dx         /label_scale ;
x1end = (x_beg*dx + output.shape[1]*scalex*dx)/label_scale ;
    
z1beg =  z_beg*dz         /label_scale ;
z1end = (z_beg*dz + output.shape[2]*scalez*dz)/label_scale ;  


## 0 degree
eps_path_2D=eps_path + "./psf-2D-eps/";
P.mkdir(eps_path_2D)


for ig_id in range(0, len(sort_angle_list)):

    ig = sort_angle_list[ig_id]
    clip1_scale=clip1_scale_list[ig_id]
    clip2_scale=clip2_scale_list[ig_id]
    

    angle = inv_angle_start + ig*dangle
    print("ig={}, angle={}".format(ig, angle))
    
    plot_2d = output[ig, :, :]
    clip = max(np.abs(plot_2d.min()), np.abs(plot_2d.max()))
    output_name=eps_path_2D+"psf-angle-"+str(angle)+".eps"
    
    P.imshow1(plot_2d.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end , x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=clip1_scale,  vmax=clip2_scale, colorbar=True, colorbar_label="Relative amplitude", cmap="gray", figsize=figsize_list[0], output_name=output_name, eps_dpi=300, title="", xtop=True, pltshow=False);
    
    output_name=eps_path_2D+"psf-angle-"+str(angle)+".bin"
    P.fwrite_file(output_name, plot_2d);
    
#################################single psf for comparsion
#################################single psf for comparsion
#################################single psf for comparsion
#################################single psf for comparsion
#################################single psf for comparsion
#################################single psf for comparsion
from scipy.fft import ifftn
from scipy.fft import fftshift
import numpy as np
from scipy.fft import ifftn
from scipy.fft import fftshift



for ig_id in range(0, len(sort_angle_list)):

    ig = sort_angle_list[ig_id] 

    angle = inv_angle_start + ig*dangle
    print("ig={}, angle={}".format(ig, angle))
    
    plot_2d = output[ig, :, :]
    [psf_nx,psf_nz] = plot_2d.shape
    
    single_psf_path = eps_path_2D + str(angle) + "/";
    P.mkdir(single_psf_path);
    

    #########################
    colorbar_label="Relative amplitude" ;
    xlabel="Distance (m)" ;
    ylabel="Depth (m)" ;
    sort_x_middle = np.int(psf_nx/wx//2*wx)
    sort_z_middle = np.int(psf_nz/wz//2*wz)
    x_arr=[wx*2-wx//2,  sort_x_middle-wx//2,  psf_nx-wx-wx//2]
    z_arr=[wz*2-wz//2,  sort_z_middle-wz//2,  psf_nz-wz-wz//2]
    # color_arr=['gray', "seismic"]
    color_arr=['gray']
    
    dkx    = 1.0/(wx*dx/1000.0);
    dkz    = 1.0/(wz*dz/1000.0);
    kx_max = 35
    kz_max = 35
    kx_num = int(kx_max/dkx);
    kz_num = int(kz_max/dkz); 
    
    for i in range(0, len(x_arr)):
        sort_x=np.int(x_arr[i])
        sort_z=np.int(z_arr[i])
        name=str(sort_x) + "-" + str(sort_z) +  "-"
    
    
###cc psf spatial domain      
        cc_single_psf = plot_2d[sort_x-wx//2: sort_x+wx//2, sort_z-wz//2: sort_z+wz//2]
        
        vmin=-0.15 * np.max(np.abs(cc_single_psf ));
        vmax=+0.15 * np.max(np.abs(cc_single_psf ))
        for ix, color in enumerate( color_arr ):
        
            output_name=single_psf_path + name + "-single-psf-" + color+".eps"
            
            P.imshow1(cc_single_psf.T,  x1beg=0, x1end=wx*dx, d1num=100, x2beg=wz*dz, x2end=0, d2num=100, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=False, colorbar_label=colorbar_label, cmap=color, figsize=(5,5), output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10));
    
    ###ex psf kx kz      
        in_arr_fft = ifftn(cc_single_psf.T);
        in_arr_fft_abs = fftshift ( np.abs(in_arr_fft) )
    
        in_arr_kxkz    = in_arr_fft_abs[wz//2 - kz_num : wz//2 + kz_num,  wx//2 - kx_num : wx//2 + kx_num]
        # in_arr_kxkz    = in_arr_fft_abs[:,:]
        in_arr_kxkz    = in_arr_kxkz/np.max(in_arr_kxkz);
        
        for ix, color in enumerate( ["seismic"] ):
            output_name=single_psf_path + name + "-kxkz-single-psf-" + color+".eps"
            
            P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=0, vmax=1, colorbar=True, colorbar_label="Normalized amplitude", cmap=color, figsize=(6, 5), output_name=output_name, eps_dpi=300, title="", xtop=True, pltshow=False);