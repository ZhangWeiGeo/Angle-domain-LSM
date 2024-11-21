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

#graben
sort_angle_list  = [15, 25, 35, 45, 55, 65, 75]
clip1_scale_list = [-3*10e-6, -3*10e-6, -3*10e-6, -3*10e-6, -3*10e-6, -3*10e-6, -3*10e-6, -3*10e-6, -3*10e-6, -3*10e-6]
clip2_scale_list = [+3*10e-6, +3*10e-6, +3*10e-6, +3*10e-6, +3*10e-6, +3*10e-6, +3*10e-6, +3*10e-6, +3*10e-6, +3*10e-6]
colorbar_ticks=((), (), (), (), (), (), (), (), (), (), (), (), (), (), ())

# P.imshow1(psf1_arr[:,:,20]);
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

# nx = x_end - x_beg
# nz = z_end - z_beg
x1beg =  x_beg*dx         /label_scale ;
x1end = (x_beg*dx + psf_nx*scalex*dx)/label_scale ;
    
z1beg =  z_beg*dz         /label_scale ;
z1end = (z_beg*dz + psf_nz*scalez*dz)/label_scale ;  


## 0 degree
eps_path_2D=eps_path + "./psf-2D-cc-de-eps/";
P.mkdir(eps_path_2D)


sort_x_middle = np.int(psf_nx/wx//2*wx)
sort_z_middle = np.int(psf_nz/wz//2*wz)
# x_arr=[wx*2-wx//2-1,  sort_x_middle-wx//2-1,  psf_nx-wx-wx//2-1]
# z_arr=[wz*2-wz//2-1,  sort_z_middle-wz//2-1,  psf_nz-wz-wz//2-1]
x_arr=[wx*2-wx//2,  sort_x_middle-wx//2,  psf_nx-wx-wx//2]
z_arr=[wz*2-wz//2,  sort_z_middle-wz//2,  psf_nz-wz-wz//2]

for ig_id in range(0, len(sort_angle_list)):

    ig = sort_angle_list[ig_id]
    angle = inv_angle_start + ig*dangle
    print("ig={}, angle={}".format(ig, angle))

################## xxxxxxxxxx   
    for ix in range(0, len(x_arr)):
        sort_x = x_arr[ix ]
        
        sort1_arr = psf1_arr[ig, sort_x, :]
        sort1_arr = sort1_arr/np.max(np.abs(sort1_arr));
        
        sort2_arr = psf2_arr[ig, sort_x, :]
        sort2_arr = sort2_arr/np.max(np.abs(sort2_arr));
        
        output_name=eps_path_2D +"x-"+str(sort_x) +"-angle-"+str(angle)  + ".eps"
        
        input_array_list = [sort1_arr, sort2_arr]
        
##################  zzzzzzzzzz        
        P.plot_graph(input_array_list, plot_number=2, dz=0.001*dz, x1beg=z1beg, x1end=z1end, d1num=0, x2beg=-0.6, x2end=1.4, d2num=0.4, label1="Depth (km)", label2="Normalized Amplitude", figsize=(6, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=("--", "--", "-"), grid_linewidth=0, line_color=("b", "r", ), fontsize=(10, 10), FontSize=(10, 10), legend_size=(10, 10), legend_position="upper right", legend_name=("CC-PSFs", "DE-PSFs", "Residual"), eps_dpi=300, output_name=output_name, plot_end=0, pltshow=False); 
        
    for iz in range(0, len(z_arr)):
        sort_z = z_arr[iz ]
        
        sort1_arr = psf1_arr[ig, :, sort_z]
        sort1_arr = sort1_arr/np.max(np.abs(sort1_arr));
        
        sort2_arr = psf2_arr[ig, :, sort_z]
        sort2_arr = sort2_arr/np.max(np.abs(sort2_arr));
        
        output_name=eps_path_2D +"z-"+str(sort_z) +"-angle-"+str(angle)  + ".eps"
        
        input_array_list = [sort1_arr, sort2_arr]
        
       
        P.plot_graph(input_array_list, plot_number=2, dz=0.001*dz, x1beg=z1beg, x1end=z1end, d1num=0, x2beg=-0.6, x2end=1.4, d2num=0.4, label1="Distance (km)", label2="Normalized Amplitude", figsize=(6, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=("--", "--", "-"), grid_linewidth=0, line_color=("b", "r", ), fontsize=(10, 10), FontSize=(10, 10), legend_size=(10, 10), legend_position="upper right", legend_name=("CC-PSFs", "DE-PSFs", "Residual"), eps_dpi=300, output_name=output_name, plot_end=0, pltshow=False); 
        
##################  xz and xz          
    for ix in range(0, len(x_arr)):
        sort_x = x_arr[ix]
        sort_z = z_arr[ix]
        
        sort1_arr = psf1_arr[ig, sort_x-wx//2: sort_x+wx//2, sort_z]
        sort1_arr = sort1_arr/np.max(np.abs(sort1_arr));
        sort2_arr = psf2_arr[ig, sort_x-wx//2: sort_x+wx//2, sort_z]
        sort2_arr = sort2_arr/np.max(np.abs(sort2_arr));
        
        output_name=eps_path_2D +"x-"+str(sort_x) +  "-z-"+str(sort_z)  +"-angle-"+str(angle)  + ".eps"
        
        input_array_list = [sort1_arr, sort2_arr]
        
       
        P.plot_graph(input_array_list, plot_number=2, dz=0.001*dz, x1beg=0, x1end=0.001*wz, d1num=0, x2beg=-0.6, x2end=1.4, d2num=0.4, label1="Distance (km)", label2="Normalized Amplitude", figsize=(4, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=("--", "--", "-"), grid_linewidth=0, line_color=("b", "r", ), fontsize=(10, 10), FontSize=(10, 10), legend_size=(10, 10), legend_position="upper right", legend_name=("CC-PSFs", "DE-PSFs", "Residual"), eps_dpi=300, output_name=output_name, plot_end=0, pltshow=False); 
        
    for iz in range(0, len(z_arr)):
        sort_x = x_arr[iz]
        sort_z = z_arr[iz]
        
        sort1_arr = psf1_arr[ig, sort_x, sort_z-wz//2: sort_z+wz//2] 
        sort1_arr = sort1_arr/np.max(np.abs(sort1_arr));
        sort2_arr = psf2_arr[ig, sort_x, sort_z-wz//2: sort_z+wz//2]
        sort2_arr = sort2_arr/np.max(np.abs(sort2_arr));
        
        output_name=eps_path_2D +"z-"+str(sort_z) +  "-x-"+str(sort_x)  +"-angle-"+str(angle)  + ".eps"
        
        input_array_list = [sort1_arr, sort2_arr]
        
       
        P.plot_graph(input_array_list, plot_number=2, dz=0.001*dz, x1beg=0, x1end=0.001*wz, d1num=0, x2beg=-0.8, x2end=1.4, d2num=0.4, label1="Depth (km)", label2="Normalized Amplitude", figsize=(4, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=("--", "--", "-"), grid_linewidth=0, line_color=("b", "r", ), fontsize=(10, 10), FontSize=(10, 10), legend_size=(10, 10), legend_position="upper right", legend_name=("CC-PSFs", "DE-PSFs", "Residual"), eps_dpi=300, output_name=output_name, plot_end=0, pltshow=False);
        
     
###################################################################copy from admm compare graben       
###################################################################copy from admm compare graben        
###AVA sort
#0:no normalized, 1:max normalzied, 2: mean normalzied, 3: normalized with 0 angle amplitude
## [wx*2-wx//2,  sort_x_middle-wx//2,  psf_nx-wx-wx//2]
x_arr=np.arange(wx*1-wx//2, psf_nx-wx//2, wx);
z_arr=np.arange(wz*1-wz//2, psf_nz-wz//2, wz);

sort_normalized   = 3; 
sort_width=3; nsmooth=0;  plot_number=3;  min_ref_value=0.005; ##sort how much ref value

for ix in range(0, len(x_arr)):
    for iz in range(0, len(z_arr)):
        sort_x = x_arr[ix]
        sort_z = z_arr[iz]
    
        sort1_arr = psf1_arr[:, sort_x, sort_z-sort_width:sort_z+sort_width+1] 
        sort2_arr = psf2_arr[:, sort_x, sort_z-sort_width:sort_z+sort_width+1] 
        
        sort1_arr = np.max(np.abs(sort1_arr), axis=1);
        sort2_arr = np.max(np.abs(sort2_arr), axis=1);
        
        sort1_arr = sort1_arr/np.max((sort1_arr));
        sort2_arr = sort2_arr/np.max((sort2_arr));
        
        output_name=eps_path_2D +  "angle-" + "x-"+ str(sort_x) +  "-z-"+str(sort_z) +  ".eps"
        
        input_array_list = [sort1_arr, sort2_arr]
        
        P.plot_graph(input_array_list, plot_number=2, dz=dangle, x1beg=inv_angle_start, x1end=inv_angle_end, d1num=0, x2beg=-0.8, x2end=1.4, d2num=0.4, label1="Reflection angle(Degree)", label2="Normalized Amplitude", figsize=(6, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=("--", "--", "-"), grid_linewidth=0, line_color=("b", "r", ), fontsize=(10, 10), FontSize=(10, 10), legend_size=(10, 10), legend_position="upper right", legend_name=("CC-PSFs", "DE-PSFs", "Residual"), eps_dpi=300, output_name=output_name, plot_end=0, pltshow=False);