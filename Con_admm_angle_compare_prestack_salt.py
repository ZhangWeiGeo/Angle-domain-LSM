#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:18:22 2023

@author: zhangjiwei
"""
# import sys;    sys.path.append("../");    from func.DEM_func import *;

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
    
    P.imshow1(input_arr.T,  x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=(6,3.5), output_name=eps_name + ".eps", eps_dpi=eps_dpi, title=title, xtop=True, pltshow=False);
    
    P.fwrite_file(data_name, input_arr);
    
# input data from path
# psf_arr = np.zeros( (nx, nz, angle_num, wangle), dtype=np.float32 );
mig_arr = np.zeros( (angle_num, nx, nz), dtype=np.float32 );
ref_arr = np.zeros( (nx, nz), dtype=np.float32 );
vel_arr = np.zeros( (nx, nz), dtype=np.float32 );
den_arr = np.zeros( (nx, nz), dtype=np.float32 );

# for i in range(0, len(psf_path)):   
#     P.fread_file_3d(psf_path[i], psf_arr[:,:,:,i], angle_num, nx, nz);
    
P.fread_file(mig_path, mig_arr, (angle_num, nx, nz));
# P.fread_file_2d(ref_path, ref_arr, nx, nz);
P.fread_file(vel_path, vel_arr, (nx, nz));
P.fread_file(den_path, den_arr, (nx, nz));

# psf_arr = psf_arr.T;
# mig_arr = mig_arr.T;
# ref_arr = ref_arr.T;
# vel_arr = vel_arr.T;
# den_arr = den_arr.T;

angle_ref_arr = C.cal_angle_ref_func(vel_arr, den_arr, angle_start=angle_start, angle_num=angle_num, dangle=dangle);

# psf_arr         = psf_arr[:, id_angle_beg:id_angle_end, x_beg:x_end, z_beg:z_end];
mig_arr         = mig_arr[id_angle_beg:id_angle_end,    x_beg:x_end, z_beg:z_end];
angle_ref_arr   = angle_ref_arr[id_angle_beg:id_angle_end, x_beg:x_end, z_beg:z_end];
ref_arr         = ref_arr[x_beg:x_end, z_beg:z_end];
vel_arr         = vel_arr[x_beg:x_end, z_beg:z_end];
den_arr         = den_arr[x_beg:x_end, z_beg:z_end];
[angle_num, nx, nz]    = mig_arr.shape





#used for compare AVA curves
print("inv_angle_start=", inv_angle_start);print("inv_angle_end =",inv_angle_end);
print("id_angle_beg=", id_angle_beg);print("id_angle_end =",id_angle_end);

##step1:
###vertical profile
sort_num            = 8;  sort_x_beg=x_beg+100; sort_x_interval=100;
vertical_sort_x_list1=np.arange(sort_x_beg, sort_x_beg+sort_num*sort_x_interval, sort_x_interval);

sort_num            = 8;  sort_x_beg=x_beg+100; sort_x_interval=100;
vertical_sort_x_list2=np.arange(sort_x_beg, sort_x_beg+sort_num*sort_x_interval, sort_x_interval);

sort_num            = 8;  sort_x_beg=x_beg+15; sort_x_interval=100;
vertical_sort_x_list3=np.arange(sort_x_beg, sort_x_beg+sort_num*sort_x_interval, sort_x_interval);

sort_num            = 8;  sort_x_beg=x_beg+20; sort_x_interval=100;
vertical_sort_x_list4=np.arange(sort_x_beg, sort_x_beg+sort_num*sort_x_interval, sort_x_interval);

# vertical_sort_x_list=np.concatenate((vertical_sort_x_list1, vertical_sort_x_list2, vertical_sort_x_list3, vertical_sort_x_list4))
vertical_sort_x_list = vertical_sort_x_list1
print("vertical_sort_x_list is", vertical_sort_x_list);
# vertical_sort_x_list=[400, 550, 600, 800]; 



###AVA sort
#0:no normalized, 1:max normalzied, 2: mean normalzied, 3: normalized with 0 angle amplitude
sort_normalized   = 3; 


sort_width=3; nsmooth=0;  plot_number=3;  min_ref_value=0.005; ##sort how much ref value

##step2:
angle_sort_x_list=vertical_sort_x_list
# angle_sort_x_list=[400, 550, 600, 800]; 


##step3:
angle_sort_z_list=[]
# angle_sort_z_list=[154, 175, 254]


##step4:
# plot_beg_list  =[-40, -40, -40]
# plot_end_list  =[+40, +40, +40]
plot_beg_list  =[-40, -40, -40]
plot_end_list  =[+40, +40, +40]

print("plot_beg_list is ",  plot_beg_list);
print("plot_end_list is ",  plot_end_list);

##id number
angle_beg_id_arr = ( ( np.asarray(plot_beg_list) -inv_angle_start) / dangle ).astype(np.int32);
angle_end_id_arr = ( ( np.asarray(plot_end_list) -inv_angle_start) / dangle ).astype(np.int32);

print("angle_beg_id_arr is ", angle_beg_id_arr);
print("angle_end_id_arr is ", angle_end_id_arr);


##step5: vertical profiles at sort_x and sort_angle;
# sort_angle_list=[45, 50, 55, 60, 65, 70, 75, 180]; ##true_angle=inv_angle_start + 5*dangle; ##

sort_angle_list =np.arange(0, 80, 5); 
sort_angle_list = np.insert(sort_angle_list, 0, 180)

# angle_2d_list = [0, 15, 30, 90]
sort_angle_list = [90, 0, 10, 20, 30, 40]
    
mig_clip1_list=[-1.5*10e-2, -2.5*10e-4, -2.5*10e-4]
mig_clip2_list=[+1.5*10e-2, +2.5*10e-4, +2.5*10e-4]
    
inv_clip1_list=[-1.5*10e-1, -2.5*10e-3, -2.5*10e-3]
inv_clip2_list=[+1.5*10e-1, +2.5*10e-3, +2.5*10e-3]
#5, 15, 25, 35, 45, 55, 65, 75, 85

###################
###################
iter_num=250
iter_num_arr=(499,) #, 150, 250, 350, )

for iter_ in range(0, len(iter_num_arr)):
    
    mig = 1.0*mig_arr;
    
    iter_num=iter_num_arr[iter_]
    
    inv = np.zeros((angle_num,nx,nz));
    name="inv-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_num)
    file_name = data_path + "admm/" + name + ".bin"
    P.fread_file(file_name, inv, (angle_num, nx, nz));
    # inv = inv.T 
    
    mig_sys = np.zeros((angle_num,nx,nz));
    # name="mig-sys-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz)
    # file_name = data_path + "ini/" + name + ".bin"
    # P.fread_file(file_name, mig_sys, (angle_num, nx, nz));
    name="mig-sys-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_num)
    file_name = data_path + "admm/" + name + ".bin"
    P.fread_file(file_name, mig_sys, (angle_num, nx, nz));
    # mig_sys = mig_sys.T
    
    
    ##mig, inv, 
    from scipy.fft import ifftn
    from scipy.fft import fftshift
    import numpy as np
    from scipy.fft import ifftn
    from scipy.fft import fftshift
    
    
    
    label1 = ""
    label2 = ""
    kx_max = 25
    kz_max = 25
    clip1  = 0.0 
    clip2  = 0.8
    legend = True
    units  = "Normalized amplitude"
    dkx    = 1.0/(nx*dx*0.001);
    dkz    = 1.0/(nz*dz*0.001);
    kx_num = int(kx_max/dkx);
    kz_num = int(kz_max/dkz);
    figsize1 = (7, 5)
    color_arr=["seismic"] #"gist_rainbow", 'gray', "summer"
    
    ## 0 degree
    eps_path_2D=eps_path + "iter-" + str(iter_num) +  "-2D-eps/";
    P.mkdir(eps_path_2D)
    data_path_2D=data_path + "iter-" + str(iter_num) +  "-2D-data/";
    P.mkdir(data_path_2D)
    
    
    for idx in range(0, len(sort_angle_list) ):
        
        if idx < len(mig_clip1_list):
            mig_clip1=mig_clip1_list[idx]
            mig_clip2=mig_clip2_list[idx]
            
            inv_clip1=inv_clip1_list[idx]
            inv_clip2=inv_clip2_list[idx]
        else:
            mig_clip1=mig_clip1_list[-1]
            mig_clip2=mig_clip2_list[-1]
            
            inv_clip1=inv_clip1_list[-1]
            inv_clip2=inv_clip2_list[-1]
        
        # angle_id = sort_angle_list[idx]
        # angle  = int(angle_id*dangle + inv_angle_start); ##### note  dangle
        
        angle = sort_angle_list[idx]
        angle_id  = int( (angle-inv_angle_start)/dangle )
        
        ###mig
        if angle_id < mig.shape[0]:
            output = mig[angle_id,:,:].reshape(nx,nz);
        else:
            output = np.sum(mig, axis=0);
            angle  = 90
        
        file = P.return_bin_name(output)
        eps_name   =  eps_path_2D  + file + "-mig-angle-" + str(angle)
        data_name  =  data_path_2D + file + "-mig-angle-" + str(angle) +  ".bin"
        imshow_and_write_km(output, eps_name, data_name, full_model=False, velocity=False, plot_min=mig_clip1, plot_max=mig_clip2);
        
        in_arr_fft      = ifftn(output.T);
        in_arr_fft_abs  = fftshift ( np.abs(in_arr_fft) )
    
        in_arr_kxkz     = in_arr_fft_abs[nz//2 - kz_num : nz//2 + kz_num,  nx//2 - kx_num : nx//2 + kx_num]
        in_arr_kxkz     = in_arr_kxkz/np.max(in_arr_kxkz);
    
        for ix, color in enumerate( color_arr ):
            output_name = eps_name + "-" + color+ "-kxkz.eps"
            
            P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
            
        ###mig_sys
        if angle_id < mig.shape[0]:
            output = mig_sys[angle_id,:,:].reshape(nx,nz);
        else:
            output = np.sum(mig_sys, axis=0);
            angle  = 90
        
        file = P.return_bin_name(output)
        eps_name   =  eps_path_2D  + file + "-sys-angle-" + str(angle)
        data_name  =  data_path_2D + file + "-sys-angle-" + str(angle) +  ".bin"
        imshow_and_write_km(output, eps_name, data_name, full_model=False, velocity=False, plot_min=mig_clip1, plot_max=mig_clip2);
        
        in_arr_fft      = ifftn(output.T);
        in_arr_fft_abs  = fftshift ( np.abs(in_arr_fft) )
    
        in_arr_kxkz     = in_arr_fft_abs[nz//2 - kz_num : nz//2 + kz_num,  nx//2 - kx_num : nx//2 + kx_num]
        in_arr_kxkz     = in_arr_kxkz/np.max(in_arr_kxkz);
    
        for ix, color in enumerate( color_arr ):
            output_name = eps_name + "-" + color+ "-kxkz.eps"
            
            P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
            
            
        ###mig_res
        mig_res = mig - mig_sys
        if angle_id < mig.shape[0]:
            output = mig_res[angle_id,:,:].reshape(nx,nz);
        else:
            output = np.sum(mig_res, axis=0);
            angle  = 90
        
        file = P.return_bin_name(output)
        eps_name   =  eps_path_2D  + file + "-res-angle-" + str(angle)
        data_name  =  data_path_2D + file + "-res-angle-" + str(angle) +  ".bin"
        imshow_and_write_km(output, eps_name, data_name, full_model=False, velocity=False, plot_min=mig_clip1, plot_max=mig_clip2);
        
        
            
            
        ###inv
        if angle_id < inv.shape[0]:
            output = inv[angle_id,:,:].reshape(nx,nz);
        else:
            output = np.sum(inv, axis=0);
            
        file = P.return_bin_name(output)
        eps_name   =  eps_path_2D  + file + "-inv-angle-" + str(angle)
        data_name  =  data_path_2D + file + "-inv-angle-" + str(angle) +  ".bin"
        imshow_and_write_km(output, eps_name, data_name, full_model=False, velocity=False, plot_min=inv_clip1, plot_max=inv_clip2);
        
        in_arr_fft = ifftn(output.T);
        in_arr_fft_abs = fftshift ( np.abs(in_arr_fft) )
    
        in_arr_kxkz    = in_arr_fft_abs[nz//2 - kz_num : nz//2 + kz_num,  nx//2 - kx_num : nx//2 + kx_num]
        in_arr_kxkz    = in_arr_kxkz/np.max(in_arr_kxkz);
    
        for ix, color in enumerate( color_arr ):
            output_name = eps_name + "-" + color+ "-kxkz.eps"
            
            P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
