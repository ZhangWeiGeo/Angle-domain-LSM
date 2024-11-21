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
sort_num            = 9;  sort_x_beg=x_beg+10; sort_x_interval=50;

vertical_sort_x_list=np.arange(sort_x_beg, sort_x_beg+sort_num*sort_x_interval, sort_x_interval);
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
plot_beg_list  =[-45, -10, -20]
plot_end_list  =[+45, +20, +10]

print("plot_beg_list is ",  plot_beg_list);
print("plot_end_list is ",  plot_end_list);

##id number
angle_beg_id_arr = ( ( np.asarray(plot_beg_list) -inv_angle_start) / dangle ).astype(np.int32);
angle_end_id_arr = ( ( np.asarray(plot_end_list) -inv_angle_start) / dangle ).astype(np.int32);

print("angle_beg_id_arr is ", angle_beg_id_arr);
print("angle_end_id_arr is ", angle_end_id_arr);


##step5: vertical profiles at sort_x and sort_angle;
sort_angle_list=[45, 55, 65, 75, 85]; ##true_angle=inv_angle_start + 5*dangle; ##
#5, 15, 25, 35, 45, 55, 65, 75, 85

###################
###################
iter_num=124
mig = 1.0*mig_arr;
inv = np.zeros((angle_num,nx,nz));
name="inv-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_num)
file_name = data_path + "admm/" + name + ".bin"
P.fread_file(file_name, inv, (angle_num, nx, nz));
# inv = inv.T 

mig_sys = np.zeros((angle_num,nx,nz));
name="mig-sys-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz)
file_name = data_path + "ini/" + name + ".bin"
P.fread_file(file_name, mig_sys, (angle_num, nx, nz));
# mig_sys = mig_sys.T


###################
###################
iter_num_arr=(62, 124, 186, 248)
clip1_arr=(+0.6, +0.6, +0.6, +0.6)
clip2_arr=(+0.6, +0.6, +0.6, +0.6)
for iter_ in range(0, len(iter_num_arr)):
    
    iter_num=iter_num_arr[iter_]
    clip1=clip1_arr[iter_]
    clip2=clip2_arr[iter_]
    
    mig = 1.0*mig_arr;
    inv = np.zeros((angle_num,nx,nz));
    name="inv-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_num)
    file_name = data_path + "admm/" + name + ".bin"
    P.fread_file(file_name, inv, (angle_num, nx, nz));
    # inv = inv.T 
    
    mig_sys = np.zeros((angle_num,nx,nz));
    name="mig-sys-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + "ini/" + name + ".bin"
    P.fread_file(file_name, mig_sys, (angle_num, nx, nz));
    # mig_sys = mig_sys.T
    
    
    ##mig, inv, 
    from scipy.fft import ifftn
    from scipy.fft import fftshift
    import numpy as np
    from scipy.fft import ifftn
    from scipy.fft import fftshift
    
    
    
    sort_num = len(vertical_sort_x_list)
    
    ## 0 degree
    eps_path_2D=eps_path + "iter-" + str(iter_num)+ "-sort_num-" + str(sort_num) +  "-2D-adcigs-eps/";
    P.mkdir(eps_path_2D)
    data_path_2D=data_path + "iter-" + str(iter_num) +  "-2D-adcigs-data/";
    P.mkdir(data_path_2D)
    
    mig_adcigs=np.zeros((angle_num,sort_num, nz), dtype=np.float32)
    inv_adcigs=np.zeros((angle_num,sort_num, nz), dtype=np.float32)
    
    for ix in range(0, sort_num):
        sort_x = vertical_sort_x_list[ix] - x_beg
        mig_adcigs[:, ix, :] = mig[:,sort_x,:];
        inv_adcigs[:, ix, :] = inv[:,sort_x,:];
        
    
    ################plot
    x1beg= vertical_sort_x_list[0]*dx/1000.0 ;
    x1end= vertical_sort_x_list[-1]*dx/1000.0 ;
    
    xtick_lables    = list(np.asarray(vertical_sort_x_list)*dx/1000.0)
    xtick_lables    = [str(num) for num in xtick_lables]
    xtick_positions = list(np.arange(int(angle_num/2), int(angle_num/2)+sort_num*angle_num, angle_num));
    
    
    x2beg= z_beg*dz                  /1000.0 ;
    x2end=(z_beg*dz + nz*dz)         /1000.0 ;
    
    xlabel="Distance (km)" ;
    ylabel="Depth (km)" ;
    
    
    
    ################plot mig
    mig_adcigs = mig_adcigs.transpose(2,1,0) 
    # mig_adcigs = np.flip(mig_adcigs, axis=2) #########for streamer data
    mig_adcigs = mig_adcigs.reshape(nz, angle_num*sort_num);
    clip = max(np.abs(mig_adcigs.min()), np.abs(mig_adcigs.max()))
    
    P.imshow1(mig_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=-clip1*clip,  vmax=clip2*clip, colorbar=True, colorbar_label="Relative amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"mig_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables );
    
    ###normalized
    mig_adcigs=mig_adcigs/((np.abs(mig_adcigs)).max()) * 0.2 / clip2;
    P.imshow1(mig_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=-0.20,  vmax=+0.20, colorbar=True, colorbar_label="Normalized amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"mig_adcigs_normalized.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables );
    
    ################plot inv
    inv_adcigs = inv_adcigs.transpose(2,1,0)
    # inv_adcigs = np.flip(inv_adcigs, axis=2) #########for streamer data
    inv_adcigs = inv_adcigs.reshape(nz, angle_num*sort_num);
    clip = max(np.abs(inv_adcigs.min()), np.abs(inv_adcigs.max()))
    
    P.imshow1(inv_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=-1.0*clip1*clip,  vmax=clip2*clip, colorbar=True, colorbar_label="Relative amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"inv_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables);
    
    ###normalized
    inv_adcigs=inv_adcigs/((np.abs(inv_adcigs)).max()) * 0.2 / clip2;
    P.imshow1(inv_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=-0.20,  vmax=+0.20, colorbar=True, colorbar_label="Normalized amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"inv_adcigs_normalized.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables);