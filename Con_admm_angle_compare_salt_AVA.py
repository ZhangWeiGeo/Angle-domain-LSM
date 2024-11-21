#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:18:22 2023

@author: zhangjiwei
"""
# import sys;    sys.path.append("../");    from func.DEM_func import *;

from Library import *
from Para import *

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

sort_num            = 9;  sort_x_beg=x_beg+100; sort_x_interval=100;
vertical_sort_x_list2=np.arange(sort_x_beg, sort_x_beg+sort_num*sort_x_interval, sort_x_interval);

sort_num            = 9;  sort_x_beg=x_beg+15; sort_x_interval=100;
vertical_sort_x_list3=np.arange(sort_x_beg, sort_x_beg+sort_num*sort_x_interval, sort_x_interval);

sort_num            = 9;  sort_x_beg=x_beg+20; sort_x_interval=100;
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
# angle_sort_x_list=[350, 400, 550, 600, 800, 850]; 
# angle_sort_x_list=[350, 450, 550, 650, 750, 850]; 

##step3:
angle_sort_z_list=[]
# angle_sort_z_list=[200, 154, 175, 254, 210]


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
# sort_angle_list=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]; ##true_angle=inv_angle_start + 5*dangle; ##
sort_angle_list = np.arange(0, 80, 5)
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
    name="mig-sys-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + "ini/" + name + ".bin"
    # P.fread_file(file_name, mig_sys, (angle_num, nx, nz));
    # mig_sys = mig_sys.T
    
    
    path = eps_path + str("AVA-width-")  + str(sort_width) + "-" + str(iter_num) + "/"
    P.mkdir(path)
    ##AVA curve
    for ix in range(0, len(angle_sort_x_list)):
        
        angle_sort_x= angle_sort_x_list[ix]
        sort_x      = angle_sort_x - x_beg
        
        ref_zero_angle_id = int((0-inv_angle_start)/dangle);
        print("ref_zero_angle_id for judge value of ref is ", ref_zero_angle_id);
        
        if sort_x<nx:
    
            if len(angle_sort_z_list)==0:
                sort_z_list=[]
                
                for iz in range(0, nz):
                    if np.fabs(angle_ref_arr[ref_zero_angle_id, sort_x, iz]) > min_ref_value:
                        sort_z_list.append(iz);
            else:
                sort_z_list = angle_sort_z_list
            
            for iz in range(len(sort_z_list)):
                
                if iz <len(angle_beg_id_arr):
                    angle_beg   = angle_beg_id_arr[iz]
                if iz <len(angle_end_id_arr):
                    angle_end   = angle_end_id_arr[iz]
                if iz <len(plot_beg_list):    
                    plot_beg    = plot_beg_list[iz]
                if iz <len(plot_end_list):   
                    plot_end    = plot_end_list[iz]
                
                
                angle_sort_z= sort_z_list[iz]
                sort_z      = angle_sort_z   ###there is no need to substract z_beg
                
                if (sort_z+sort_width+1) < nz and sort_z-sort_width >0:
                    tmp1 = angle_ref_arr[angle_beg : angle_end, sort_x, sort_z]
                    
                    tmp2 = mig[angle_beg : angle_end, sort_x, sort_z-sort_width:sort_z+sort_width+1]
                    tmp3 = inv[angle_beg : angle_end, sort_x, sort_z-sort_width:sort_z+sort_width+1]
                    tmp4 = mig_sys[angle_beg : angle_end, sort_x, sort_z-sort_width:sort_z+sort_width+1]
                    
                    # arr1 = np.max(np.abs(tmp1), axis=1);
                    arr1 = 1.0*(tmp1)
                    arr2 = np.max(np.abs(tmp2), axis=1);
                    arr3 = np.max(np.abs(tmp3), axis=1);
                    arr4 = np.max(np.abs(tmp4), axis=1);
                    
                    if nsmooth!=0:
                        arr3 = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, arr3, axis=0)
                        arr4 = filtfilt(np.ones(nsmooth)/float(nsmooth), 1, arr4, axis=0)
                    
                
                    if sort_normalized==1:
                        arr1 = 1.0*arr1
                        arr2 = arr2/np.max(np.abs(arr2[:])) *np.max(np.abs(arr1))
                        arr3 = arr3/np.max(np.abs(arr3[:])) *np.max(np.abs(arr1))
                        arr4 = arr4/np.max(np.abs(arr4[:])) *np.max(np.abs(arr1))
                    if sort_normalized==2:
                        arr1 = 1.0*arr1
                        mean = np.mean(arr1);
                        arr2 = arr2/np.max(np.abs(arr2[:]))*np.abs(mean)
                        arr3 = arr3/np.max(np.abs(arr3[:]))*np.abs(mean)
                        arr4 = arr4/np.max(np.abs(arr4[:]))*np.abs(mean)
                    if sort_normalized==3:
                        zero_angle_id = int((0-plot_beg)/dangle);
                        print("zero_angle_id is ", zero_angle_id);
                        arr1 = 1.0*arr1
                        arr2 = arr2/arr2[zero_angle_id] * arr1[zero_angle_id]
                        arr3 = arr3/arr3[zero_angle_id] * arr1[zero_angle_id]
                        arr4 = arr4/arr4[zero_angle_id] * arr1[zero_angle_id]
                        
                        
                    name = "angle-ix-" + str(sort_x+x_beg) + "-z-" + str(int(sort_z+z_beg)) + ".eps"
                    print(name);
                    input_arr_list=[]
                    input_arr_list.extend([arr1, arr2,arr3, arr4])
                    print(input_arr_list[0].shape)
                    P.plot_graph(input_arr_list, plot_number=plot_number, dz=dangle, x1beg=plot_beg, x1end=plot_end, d1num=0, label1="Reflection angle(Degree)", label2="Angle-dependent reflectivity", figsize=(6, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=('o-','D-', 's-', 's-'), grid_linewidth=0, line_color=("k", "b", "r", "g" ), fontsize=(10,10), FontSize=(10,10), legend_size=(10,10), legend_position=("best"), legend_name=("Reference", "DE-KM", "DE-IDLSM", "sys-RTM"), eps_dpi=300, output_name=path + name)
    
    
    
    
    # angle_ref_arr, net_out/inv, mig
    for ix in range(0, len(vertical_sort_x_list)):
        true_x = vertical_sort_x_list[ix]
        sort_x = true_x - x_beg;
        
        if sort_x < nx:
            path = eps_path + "admm/" + str(sort_x+x_beg)  + "-" + str(iter_num) + "/"
            P.mkdir(path)
            
            for ig_id in range(0, len(sort_angle_list)):
                ig = sort_angle_list[ig_id]
                print("ix={}, ig={}".format(ix, ig));print("ix={}, ix={}".format(ix, ig));
                
                arr1 = 1.0* angle_ref_arr[ig, sort_x, :]
                arr2 = 1.0* mig[ig, sort_x, :]
                arr3 = 1.0* inv[ig, sort_x, :]
                arr4 = 1.0* mig_sys[ig, sort_x, :]
                
                angle = inv_angle_start + ig*dangle;
                
                if sort_normalized:
                    arr1 = arr1
                    arr2 = arr2/np.max( np.abs(arr2) )*np.max( np.abs(arr1) )
                    arr3 = arr3/np.max( np.abs(arr3) )*np.max( np.abs(arr1) )
                    arr3 = arr3/np.max( np.abs(arr3) )*np.max( np.abs(arr1) )
                    arr4 = arr4/np.max( np.abs(arr4) )*np.max( np.abs(arr1) )
                    
                cc1          = np.dot(arr1, arr2)/np.sqrt(np.dot(arr1, arr1))/np.sqrt(np.dot(arr2, arr2))
                cc2          = np.dot(arr1, arr3)/np.sqrt(np.dot(arr1, arr1))/np.sqrt(np.dot(arr3, arr3))
                name1        = "DE-KM, NCC={:.3f}".format(cc1)
                name2        = "DE-IDLSM, NCC={:.3f}".format(cc2)    
                
                name = "ix-" + str(sort_x+x_beg) + "-ig-" + str(int(angle))   + ".eps"
                input_arr_list=[]
                input_arr_list.extend([arr1, arr2, arr3, arr4])
                print(input_arr_list[0].shape)
                P.plot_graph(input_arr_list, plot_number=plot_number, x1beg=z_beg*dz/1000.0, x1end=z_end*dz/1000.0, d1num=0, label1="Depth (km)", label2="Reflectivity", figsize=(6, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=("-","--","--","--"), grid_linewidth=0, line_color=("k", "b", "r", "g"), fontsize=(10,10), FontSize=(10,10), legend_size=(10,10), legend_position=("best"), legend_name=("Reference", name1, name2, "sys-RTM"), eps_dpi=300, output_name=path + name);