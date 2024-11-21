#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:20:05 2023

@author: zhangjiwei
"""


from Library import *
from make_data_3D import imshow_and_write_km as pw

def output_train_3D_eps(iter_m, deq, data_path, eps_path, log_path, net_out_sys, net_out_sys_res, net_out, ou_y, ref_arr, eps_dpi=100):
    
    [_, angle_num, nx, nz] = net_out_sys.size()
        
    mig_sys = T.torch_to_np(net_out_sys).reshape((angle_num, nx,nz));
    mig_res = T.torch_to_np(net_out_sys_res).reshape((angle_num, nx,nz));
    inv     = T.torch_to_np(net_out).reshape((angle_num, nx,nz));
        
    if iter_m==0:    
        ou_y    = T.torch_to_np(ou_y).reshape(angle_num, nx,nz);
        name= "ou-y-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz)
        file_name = data_path + name + ".bin"
        P.fwrite_file(file_name, ou_y);
        
        name="ou-y-" + str(nx) + "-"  + str(nz)
        file_name = data_path + name + ".bin"
        P.fwrite_file(file_name, np.sum(ou_y, axis=0));
        
        # #ref
        # name="ref-" + str(angle_num) + + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
        # file_name = data_path + name + ".bin"
        # P.fwrite_file(file_name, x_data);
        
        
    # inv
    name="inv-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    P.fwrite_file(file_name, inv);
    
    # inv
    name="inv-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    P.fwrite_file(file_name, np.sum(inv, axis=0));
        
        
    # mig sys
    name="mig-sys-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    P.fwrite_file(file_name, mig_sys);
      
        
    # mig res
    name="mig-res-" + str(angle_num) + "-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    P.fwrite_file(file_name, mig_res);
    
 
    # # res 1
    # name="inv-res1-" + str(angle_num) + + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    # file_name = data_path + name + ".bin"
    # P.fwrite_file(file_name, x_data);
        
    # # res 3
    # name="inv-res3-" + str(angle_num) + + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    # file_name = data_path + name + ".bin"
    # P.fwrite_file(file_name, x_data);