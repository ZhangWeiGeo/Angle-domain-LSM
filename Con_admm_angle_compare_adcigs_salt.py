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
sort_num            = 9;  sort_x_beg=x_beg+100; sort_x_interval=50;
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

wiggle_angle_beg=-30;   wiggle_angle_end=33;    wiggle_angle_interval=3

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
sort_angle_list=[45, 55, 65, 75, 85]; ##true_angle=inv_angle_start + 5*dangle; ##
#5, 15, 25, 35, 45, 55, 65, 75, 85

###################
###################
###################
###################
iter_num=250
iter_num_arr=(499,) #, 150, 250, 350, )

mig_clip1_list=[-2.5*10e-4, -2.5*10e-4]
mig_clip2_list=[+2.5*10e-4, +2.5*10e-4]
    
inv_clip1_list=[-2.5*10e-3, -2.5*10e-3]
inv_clip2_list=[+2.5*10e-3, +2.5*10e-3]

mig_nor_clip1         = -0.3;   mig_nor_clip2=0.3;
inv_nor_clip1         = -0.3;   inv_nor_clip2=0.3;

for iter_ in range(0, len(iter_num_arr)):
    
    iter_num=iter_num_arr[iter_]

    mig_clip1=mig_clip1_list[iter_]
    mig_clip2=mig_clip2_list[iter_]
    
    inv_clip1=inv_clip1_list[iter_]
    inv_clip2=inv_clip2_list[iter_]
    
    mig = 1.0*mig_arr;
    
    inv = np.zeros((angle_num,nx,nz));
    name="inv-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_num)
    file_name = data_path + "admm/" + name + ".bin"
    P.fread_file(file_name, inv, (angle_num, nx, nz));
    ## inv = inv.T 
    
    res = np.zeros((angle_num,nx,nz));
    name="mig-res-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_num)
    file_name = data_path + "admm/" + name + ".bin"
    P.fread_file(file_name, res, (angle_num, nx, nz));
    ## inv = inv.T 
    
    mig_sys = np.zeros((angle_num,nx,nz));
    # name="mig-sys-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz)
    # file_name = data_path + "ini/" + name + ".bin"
    name="mig-sys-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_num)
    file_name = data_path + "admm/" + name + ".bin"
    P.fread_file(file_name, mig_sys, (angle_num, nx, nz));
    ## mig_sys = mig_sys.T
    
    
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
    
    mig_adcigs    =np.zeros((angle_num,sort_num, nz), dtype=np.float32)
    inv_adcigs    =np.zeros((angle_num,sort_num, nz), dtype=np.float32)
    inv_res_adcigs=np.zeros((angle_num,sort_num, nz), dtype=np.float32)
    sys_adcigs    =np.zeros((angle_num,sort_num, nz), dtype=np.float32)
    
    for ix in range(0, sort_num):
        sort_x                      = vertical_sort_x_list[ix] - x_beg
        mig_adcigs[:, ix, :]        = mig[:,sort_x,:];
        inv_adcigs[:, ix, :]        = inv[:,sort_x,:];
        inv_res_adcigs[:, ix, :]    = res[:,sort_x,:];
        sys_adcigs[:, ix, :]        = mig_sys[:,sort_x,:];
        
    res_adcigs               = mig_adcigs - sys_adcigs;
    
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
    
    inv_max = max(np.abs(inv_adcigs.min()), np.abs(inv_adcigs.max()))
    mig_max = max(np.abs(mig_adcigs.min()), np.abs(mig_adcigs.max()))
    sys_max = max(np.abs(sys_adcigs.min()), np.abs(sys_adcigs.max()))
    
################plot inv
    inv_adcigs = inv_adcigs.transpose(2,1,0)
    # inv_adcigs = np.flip(inv_adcigs, axis=2) #########for streamer data
    inv_adcigs = inv_adcigs.reshape(nz, angle_num*sort_num);
    
    P.imshow1(inv_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=inv_clip1,  vmax=inv_clip2, colorbar=True, colorbar_label="Relative amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"a_inv_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables);
    
    ###normalized
    inv_adcigs=inv_adcigs/inv_max;
    P.imshow1(inv_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=inv_nor_clip1,  vmax=inv_nor_clip2, colorbar=True, colorbar_label="Normalized amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"nor_inv_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables);
    
    
################plot inv_res
    inv_res_adcigs = inv_res_adcigs.transpose(2,1,0)
    # inv_adcigs = np.flip(inv_adcigs, axis=2) #########for streamer data
    inv_res_adcigs = inv_res_adcigs.reshape(nz, angle_num*sort_num);
    
    P.imshow1(inv_res_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=mig_clip1,  vmax=mig_clip2, colorbar=True, colorbar_label="Relative amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"a_inv_res_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables);
    
    ###normalized
    inv_res_adcigs=inv_res_adcigs/mig_max;
    P.imshow1(inv_res_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=mig_nor_clip1,  vmax=mig_nor_clip2, colorbar=True, colorbar_label="Normalized amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"nor_inv_res_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables);
    
    
    ################plot mig
    if iter_==0:
        mig_adcigs = mig_adcigs.transpose(2,1,0) 
        # mig_adcigs = np.flip(mig_adcigs, axis=2) #########for streamer data
        mig_adcigs = mig_adcigs.reshape(nz, angle_num*sort_num);
        
        P.imshow1(mig_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=mig_clip1,  vmax=mig_clip2, colorbar=True, colorbar_label="Relative amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"a_mig_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables );
        
        ###normalized
        mig_adcigs=mig_adcigs/mig_max;
        P.imshow1(mig_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=mig_nor_clip1,  vmax=mig_nor_clip2, colorbar=True, colorbar_label="Normalized amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"nor_mig_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables );
    

###############plot sys adcigs
        sys_adcigs = sys_adcigs.transpose(2,1,0) 
        # sys_adcigs = np.flip(sys_adcigs, axis=2) #########for streamer data
        sys_adcigs = sys_adcigs.reshape(nz, angle_num*sort_num);
        # clip = max(np.abs(sys_adcigs.min()), np.abs(sys_adcigs.max()))
        
        P.imshow1(sys_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=mig_clip1,  vmax=mig_clip2, colorbar=True, colorbar_label="Relative amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"a_sys_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables );
    
        ###normalized
        sys_adcigs=sys_adcigs/mig_max
        P.imshow1(sys_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=mig_nor_clip1,  vmax=mig_nor_clip2, colorbar=True, colorbar_label="Normalized amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"nor_sys_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables );
        
###############plot res adcigs
        res_adcigs = res_adcigs.transpose(2,1,0)
        # sys_adcigs = np.flip(sys_adcigs, axis=2) #########for streamer data
        res_adcigs = res_adcigs.reshape(nz, angle_num*sort_num);
        
        P.imshow1(res_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=mig_clip1,  vmax=mig_clip2, colorbar=True, colorbar_label="Relative amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"a_res_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables );
    
        ###normalized
        res_adcigs=res_adcigs/mig_max ;
        P.imshow1(res_adcigs, x1beg=0, x1end=0, d1num=0, x2beg=x2end, x2end=x2beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=mig_nor_clip1,  vmax=mig_nor_clip2, colorbar=True, colorbar_label="Normalized amplitude", cmap="gray", figsize=(10,3.5), output_name=eps_path_2D+"nor_res_adcigs.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions=xtick_positions, xtick_lables=xtick_lables );


###############wiggle figure    
###############wiggle figure 
###############wiggle figure
###############wiggle figure    
###############wiggle figure    
###############wiggle figure 
###############wiggle figure
###############wiggle figure   
    for ix in range(0, sort_num):
        mig_adcigs=np.zeros((angle_num, nz), dtype=np.float32)
        inv_adcigs=np.zeros((angle_num, nz), dtype=np.float32)
        sys_adcigs=np.zeros((angle_num, nz), dtype=np.float32) 
        
        sort_x           = vertical_sort_x_list[ix] - x_beg
        mig_adcigs[:, :] = mig[:,sort_x,:];
        inv_adcigs[:, :] = inv[:,sort_x,:];
        sys_adcigs[:, :] = mig_sys[:,sort_x,:];
        res_adcigs       = mig_adcigs[:, :] - sys_adcigs[:, :]

        mig_adcigs = mig_adcigs.T/np.max(np.abs(mig_adcigs))
        inv_adcigs = inv_adcigs.T/np.max(np.abs(inv_adcigs))
        sys_adcigs = sys_adcigs.T/np.max(np.abs(sys_adcigs))
        res_adcigs = res_adcigs.T/np.max(np.abs(mig_adcigs))
        
        name1 = eps_path_2D + "x-" + str(vertical_sort_x_list[ix]) + "-mig" + ".eps"
        name2 = eps_path_2D + "x-" + str(vertical_sort_x_list[ix]) + "-inv" + ".eps"
        name3 = eps_path_2D + "x-" + str(vertical_sort_x_list[ix]) + "-sys" + ".eps"
        name4 = eps_path_2D + "x-" + str(vertical_sort_x_list[ix]) + "-res" + ".eps"
        
        beg=wiggle_angle_beg-inv_angle_start;
        end=wiggle_angle_end-inv_angle_start;
        plot_number= np.int((end-beg)//wiggle_angle_interval);
        figsize=(20, 10)

        ##plot inv
        P.plot_graph_multi(inv_adcigs[:,beg:end:wiggle_angle_interval], plot_number=plot_number, x1beg=x2beg, x1end=x2end, d1num=0, x2beg=-0.8, x2end=1, d2num=0.5, x2beg_new=wiggle_angle_beg, d2num_new=wiggle_angle_interval, d2num_new_plot=5, d2label="Angle (Degree)", label1="", label2=ylabel, axis_width=1, axis_length=1, linewidth=1, line_color=("b", ), linestyle=("-",), FontSize=(24,24), fontsize=(24,24), figsize=figsize, reverse_1=False, reverse_2=True, output_name=name2, eps_dpi=300);
    
        if iter_==0:
            ##plot mig
            P.plot_graph_multi(mig_adcigs[:,beg:end:wiggle_angle_interval], plot_number=plot_number, x1beg=x2beg, x1end=x2end, d1num=0, x2beg=-1.0, x2end=0.9, d2num=0.5, x2beg_new=wiggle_angle_beg, d2num_new=wiggle_angle_interval, d2num_new_plot=5, d2label="Angle (Degree)", label1="", label2=ylabel, axis_width=1, axis_length=1, linewidth=1, line_color=("b", ), linestyle=("-",), FontSize=(24,24), fontsize=(24,24), figsize=figsize, reverse_1=False, reverse_2=True, output_name=name1, eps_dpi=300);
        
            ##plot sys
            P.plot_graph_multi(sys_adcigs[:,beg:end:wiggle_angle_interval], plot_number=plot_number, x1beg=x2beg, x1end=x2end, d1num=0, x2beg=-1.0, x2end=0.9, d2num=0.5, x2beg_new=wiggle_angle_beg, d2num_new=wiggle_angle_interval, d2num_new_plot=5, d2label="Angle (Degree)", label1="", label2=ylabel, axis_width=1, axis_length=1, linewidth=1, line_color=("b", ), linestyle=("-",), FontSize=(24,24), fontsize=(24,24), figsize=figsize, reverse_1=False, reverse_2=True, output_name=name3, eps_dpi=300);
            ##plot res
            P.plot_graph_multi(res_adcigs[:,beg:end:wiggle_angle_interval], plot_number=plot_number, x1beg=x2beg, x1end=x2end, d1num=0, x2beg=-1.0, x2end=0.9, d2num=0.5, x2beg_new=wiggle_angle_beg, d2num_new=wiggle_angle_interval, d2num_new_plot=5, d2label="Angle (Degree)", label1="", label2=ylabel, axis_width=1, axis_length=1, linewidth=1, line_color=("b", ), linestyle=("-",), FontSize=(24,24), fontsize=(24,24), figsize=figsize, reverse_1=False, reverse_2=True, output_name=name4, eps_dpi=300);
