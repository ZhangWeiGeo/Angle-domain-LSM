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


sort_width=30; nsmooth=0;  plot_number=3;  min_ref_value=0.005; ##sort how much ref value

##step2:
angle_sort_x_list=vertical_sort_x_list
# angle_sort_x_list=[400, 550, 600, 800]; 


##step3:
angle_sort_z_list=[]
# angle_sort_z_list=[154, 175, 254]


##step4:
# plot_beg_list  =[-40, -40, -40]
# plot_end_list  =[+40, +40, +40]
plot_beg_list  =[-45, -45, -45]
plot_end_list  =[+45, +45, +45]


print("plot_beg_list is ",  plot_beg_list);
print("plot_end_list is ",  plot_end_list);

##id number
angle_beg_id_arr = ( ( np.asarray(plot_beg_list) -inv_angle_start) / dangle ).astype(np.int32);
angle_end_id_arr = ( ( np.asarray(plot_end_list) -inv_angle_start) / dangle ).astype(np.int32);

print("angle_beg_id_arr is ", angle_beg_id_arr);
print("angle_end_id_arr is ", angle_end_id_arr);


##step5: vertical profiles at sort_x and sort_angle;
sort_angle_list =np.arange(0, 80, 5); 
sort_angle_list=np.append(sort_angle_list,180)
#5, 10, 15, 20, 25, 30, 35, 40,  
# sort_angle_list=np.arange(45, 45+40, 5);
# sort_angle_list=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85];
##true_angle=inv_angle_start + 5*dangle; ##
#5, 15, 25, 35, 45, 55, 65, 75, 85

###################
###################
normalized_way=0
iter_num=250
iter_num_arr=(450,) #, 150, 250, 350, )

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
    
    
    path = eps_path + str("AVA-wavelet-width-")  + str(sort_width) + "-" + str(iter_num) + "/"
    P.mkdir(path); 
    P.mkdir(path+"compare1")
    P.mkdir(path+"compare2")
    P.mkdir(path+"compare3")
    
    
    ##AVA curve
    for ix in range(0, len(angle_sort_x_list)):
        
        angle_sort_x = angle_sort_x_list[ix]
        sort_x       = angle_sort_x - x_beg
        
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
                
                zbeg=sort_z-sort_width
                zend=sort_z+sort_width+1
                
                if zbeg<0:
                    zbeg=0
                if zend>nz:
                    zend=nz

##################vertical direction                
                tmp2 = mig[angle_beg : angle_end, sort_x, zbeg:zend]
                tmp3 = inv[angle_beg : angle_end, sort_x, zbeg:zend]
                tmp4 = mig_sys[angle_beg : angle_end, sort_x, zbeg:zend]
                    
                    
                name = "anglex-ix-" + str(sort_x+x_beg) + "-z-" + str(int(sort_z+z_beg)) + ".eps"
                
                
                sort1_list=[]
                sort2_list=[]
                sort3_list=[]
                lengend_name=[];
                
                for ig_id in range(0, len(sort_angle_list)):
            
                    ig = sort_angle_list[ig_id]
                    angle = inv_angle_start + ig*dangle
                    print("ig={}, angle={}".format(ig, angle))
            
                    if np.fabs(angle) < 90:
                        sort1_arr = tmp2[ig, :]
                        sort2_arr = tmp3[ig, :]
                        sort3_arr = tmp4[ig, :]
                    
                        sort1_list.append(sort1_arr)
                        sort2_list.append(sort2_arr)
                        sort3_list.append(sort3_arr)
                        lengend_name.append("Angle=" + str(angle));
        
    #####normalized way
                if normalized_way==0:
    # 找到列表中所有数组的最大值
                    max_value1 = np.max([np.max(np.abs(arr)) for arr in sort1_list])
                    max_value2 = np.max([np.max(np.abs(arr)) for arr in sort2_list])
                    max_value3 = np.max([np.max(np.abs(arr)) for arr in sort3_list])
    # 对每个数组进行归一化处理
                    sort1_list = [(arr / max_value1) for arr in sort1_list]
                    sort2_list = [(arr / max_value2) for arr in sort2_list]
                    sort3_list = [(arr / max_value3) for arr in sort3_list]
                else:
                    sort1_list = [arr / np.max(np.abs(arr)) for arr in sort1_list ]
                    sort2_list = [arr / np.max(np.abs(arr)) for arr in sort2_list ] 
                    sort3_list = [arr / np.max(np.abs(arr)) for arr in sort2_list ] 
    
                output_name1=path + "compare1/" + name
                output_name2=path + "compare2/" + name
                output_name3=path + "compare3/" + name
                linestyle=("-", "-", "-", "-", "--", "--", "--", "--")
                line_color=("b", "r", "g", "k", "b", "r", "g", "k")
                
                if iz ==0 or iz==1:
                    x2beg=-1.0;x2end=1.2;d2num=0.3;
                else:
                    x2beg=-1.2;x2end=1.0;d2num=0.3;
                    
                P.plot_graph(sort2_list, plot_number=len(sort2_list), dz=0.001*dz, x1beg=0, x1end=0.001*(sort_width*2+1), d1num=0, x2beg=x2beg, x2end=x2end, d2num=d2num, label1="Depth (km)", label2="Normalized Amplitude", figsize=(4, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=(9, 9), FontSize=(9, 9), legend_size=(7, 7), legend_position="upper right", legend_name=lengend_name, eps_dpi=300, output_name=output_name2, plot_end=0, pltshow=False);
                
                if iter_ ==0:
                    P.plot_graph(sort1_list, plot_number=len(sort1_list), dz=0.001*dz, x1beg=0, x1end=0.001*(sort_width*2+1), d1num=0, x2beg=x2beg, x2end=x2end, d2num=d2num, label1="Depth (km)", label2="Normalized Amplitude", figsize=(4, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=(9, 9), FontSize=(9, 9), legend_size=(7, 7), legend_position="upper right", legend_name=lengend_name, eps_dpi=300, output_name=output_name1, plot_end=0, pltshow=False);
                    
                    P.plot_graph(sort3_list, plot_number=len(sort2_list), dz=0.001*dz, x1beg=0, x1end=0.001*(sort_width*2+1), d1num=0, x2beg=x2beg, x2end=x2end, d2num=d2num, label1="Depth (km)", label2="Normalized Amplitude", figsize=(4, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=(9, 9), FontSize=(9, 9), legend_size=(7, 7), legend_position="upper right", legend_name=lengend_name, eps_dpi=300, output_name=output_name3, plot_end=0, pltshow=False);
                
                
                
##################horinzontal direction  
    #             sort_z      = angle_sort_z   ###there is no need to substract z_beg
                
    #             xbeg=sort_x-sort_width
    #             xend=sort_x+sort_width+1
                
    #             if xbeg<0:
    #                 xbeg=0
    #             if xend>nz:
    #                 xend=nx
                    
    #             tmp2 = mig[angle_beg : angle_end, xbeg:xend, sort_z]
    #             tmp3 = inv[angle_beg : angle_end, xbeg:xend, sort_z]
    #             tmp4 = mig_sys[angle_beg : angle_end, xbeg:xend, sort_z]
                    
                    
    #             name = "anglez-ix-" + str(sort_x+x_beg) + "-z-" + str(int(sort_z+z_beg)) + ".eps"
                
                
    #             sort1_list=[]
    #             sort2_list=[]
    #             sort3_list=[]
    #             lengend_name=[];
                
    #             for ig_id in range(0, len(sort_angle_list)):
            
    #                 ig = sort_angle_list[ig_id]
    #                 angle = inv_angle_start + ig*dangle
    #                 print("ig={}, angle={}".format(ig, angle))
            
    #                 if np.fabs(angle) < 90:
    #                     sort1_arr = tmp2[ig, :]
    #                     sort2_arr = tmp3[ig, :]
    #                     sort3_arr = tmp4[ig, :]
                    
    #                     sort1_list.append(sort1_arr)
    #                     sort2_list.append(sort2_arr)
    #                     sort3_list.append(sort3_arr)
    #                     lengend_name.append("Angle=" + str(angle));
        
    # #####normalized way
    #             if normalized_way==0:
    # # 找到列表中所有数组的最大值
    #                 max_value1 = np.max([np.max(np.abs(arr)) for arr in sort1_list])
    #                 max_value2 = np.max([np.max(np.abs(arr)) for arr in sort2_list])
    #                 max_value3 = np.max([np.max(np.abs(arr)) for arr in sort3_list])
    # # 对每个数组进行归一化处理
    #                 sort1_list = [(arr / max_value1) for arr in sort1_list]
    #                 sort2_list = [(arr / max_value2) for arr in sort2_list]
    #                 sort3_list = [(arr / max_value3) for arr in sort3_list]
    #             else:
    #                 sort1_list = [arr / np.max(np.abs(arr)) for arr in sort1_list ]
    #                 sort2_list = [arr / np.max(np.abs(arr)) for arr in sort2_list ] 
    #                 sort3_list = [arr / np.max(np.abs(arr)) for arr in sort2_list ] 
    
    #             output_name1=path + "compare1/" + name
    #             output_name2=path + "compare2/" + name
    #             output_name3=path + "compare3/" + name
    #             linestyle=("-", "-", "-", "-", "--", "--", "--", "--")
    #             line_color=("b", "r", "g", "k", "b", "r", "g", "k")
                    
    #             x2beg=-1.0;x2end=1.4;d2num=0.4;
                    
    #             P.plot_graph(sort1_list, plot_number=len(sort1_list), dz=0.001*dz, x1beg=0, x1end=0.001*(sort_width*2+1), d1num=0, x2beg=x2beg, x2end=x2end, d2num=d2num, label1="Distance (km)", label2="Normalized Amplitude", figsize=(4, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=(9, 9), FontSize=(9, 9), legend_size=(7, 7), legend_position="upper right", legend_name=lengend_name, eps_dpi=300, output_name=output_name1, plot_end=0, pltshow=False);
                
    #             if iter_ ==0:
    #                 P.plot_graph(sort2_list, plot_number=len(sort2_list), dz=0.001*dz, x1beg=0, x1end=0.001*(sort_width*2+1), d1num=0, x2beg=x2beg, x2end=x2end, d2num=d2num, label1="Distance (km)", label2="Normalized Amplitude", figsize=(4, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=(9, 9), FontSize=(9, 9), legend_size=(7, 7), legend_position="upper right", legend_name=lengend_name, eps_dpi=300, output_name=output_name2, plot_end=0, pltshow=False);
                    
    #             P.plot_graph(sort3_list, plot_number=len(sort2_list), dz=0.001*dz, x1beg=0, x1end=0.001*(sort_width*2+1), d1num=0, x2beg=x2beg, x2end=x2end, d2num=d2num, label1="Distance (km)", label2="Normalized Amplitude", figsize=(4, 3), axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=(9, 9), FontSize=(9, 9), legend_size=(7, 7), legend_position="upper right", legend_name=lengend_name, eps_dpi=300, output_name=output_name3, plot_end=0, pltshow=False);