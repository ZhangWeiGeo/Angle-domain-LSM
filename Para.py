#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:10:06 2023

@author: zhangjiwei
"""

import sys
import os
home_path= os.getenv("HOME"); 
# sys.path.append(home_path + "/zw_lib/python/func/")
sys.path.append("./func")
# sys.path.append("/ibex/ai/home/zhangw0c/DEM-IDLSM/R-3-19-3D/Para2-angle/graben-vp-820-315-3/");
sys.path.append(home_path + "/Para2-angle/salt-670-201-shendi-new/");
import common  as C
import plot_func as P
import numpy as np
from vel_ex_100_part2 import *

# input paramters from shell.sh
parameters_arr = sys.argv[:]; # parameters_arr = sys.argv[1:];
if len(parameters_arr) ==1:
    parameters_arr = ["Progama_name", 0, 30, 32, "False", 5e-4, 0, 0, 0, 0, 0, 0, 50, 5e-4, "True", "False", 50, 10, 0.01, 1, "/media/zhangjiwei/z16/paper/DE-AD-IDLSM-multi-reg-para/salt-670-201-shendi-new/inv/down-ex-100/l2-l2/1-1-1-5e-4-0.2-0.2-0-0-0-0-500-5e-4-12-1"] ;

parameters_log = 'parameters_arr is: {}'.format(parameters_arr); print(parameters_log);

# if or not normalize the input(x) and labels(y)
normalized_x = True;    normalized_y = True;

# network parameters
DEM_or_DU       = int( parameters_arr[1] );     #0:Deep unrolling, >1:DEM, 
Unrolling_way   = 1;                            #0:Standard GD; 1:DL-based GD; 2:DL-based Proximal GD
check_point_bool= True; net_iter_input_nor=True;

ini_inv_niter=1; fin_inv_niter=500;##L2 inverted for comparsion
if DEM_or_DU==0:
    f_net_iter = 15;      tol1 = -1e-3;         #Maxmium iteration number for forward p and tol
    b_net_iter = 15;      tol2 = -1e-3;         #Maxmium iteration number for backward p and tol
else:
    f_net_iter = 100;     tol1 =  +1e-3;          
    b_net_iter = 100;     tol2 =  +1e-3;          
    check_point_bool= False;                    #backward by myself

network_bool= int( parameters_arr[2] );       #1, 2, 3, 4, 30, 40, 50 

in_channels = 1;        ou_channels  = 1;           filter_num = int( parameters_arr[3] );
kernel_size = (3,3);    padding      = (1,1);       padding_mode = 'replicate';
is_deconv   = True;     is_batchnorm = True;        spectral_norm = True;#spectral_norm = str(parameters_arr[4]);
Leaky_value = 0.0  ;    Leaky        = 1.0 * Leaky_value;

Initial_way_bool = 0; 
init_weight      = float( parameters_arr[5] );
bias_bool1       = True;  bias_bool2 = True;

# add drop_layer number layers with fully connected network or CNN with drop_out for uncertainty quantification
neu_in_num = nx*nz;             neu_ou_num = 100;       drop_layer = 2;
drop_out      = float( parameters_arr[6] ) ;            neu_ini = 0.001;        
pred_drop_out = float( parameters_arr[7] ) ;


# Add TV loss function ?: it is neccsarry for DEM approach, even though the observed data does not contain noise
admm_bool  = 1;          tv_type      = 0;
reg_change = False;      inner_epochs = 1;      CG_bool = 1;
# TV loss parameters
alpha_1 = float( parameters_arr[8] );     landa_1 = float( parameters_arr[9] );    ephsion_1 = 1.000;
# L1 loss parameters
beta_3  = float( parameters_arr[10] );    ephsion_3 = float( parameters_arr[11] );    angle_reg_para4=float( parameters_arr[18] );      TV_2D_3D_bool=float( parameters_arr[19] ); # 0:2D, 1: 3D(y,z), 2:3D(x,y,z)

# training and predicting parameters,  set lr_bool as False or 1(CosineAnnealingLR), 2(CosineAnnealingWarmRestarts), 3(ExponentialLR), 4(StepLR)
batch_size    = 1;#uselss parameter
train_epochs  = int(parameters_arr[12]);    
output_epochs = int(train_epochs/10);    
save_model_epochs=50;

lr_bool       = 0;                        
lr_train      = float( parameters_arr[13] );
lr_max_epochs = inner_epochs;               
lr_eta_min    = lr_train/10;

# predicting number for uncertainty quantification
train_bool       = eval(str(parameters_arr[14]));####noted that there is no need to use str(parameters_arr[14])
predict_bool     = eval(str(parameters_arr[15]));####noted that there is no need to use str(parameters_arr[15])
predcit_at_epoch = int(parameters_arr[16]);
UQ_num           = int(parameters_arr[17]);
output_bool      = False ;			output_other=False;##used for debug
fig_type       = ".jpg" ;             
eps_dpi = 130;
color_arr=["gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"]

if DEM_or_DU ==0:
    output_bool_interval = 1;
else:
    output_bool_interval = 10;

# output_path
# output_path    = pwd_path + "/" + str(parameters_arr[-1]) + "/";
output_path      =                  str(parameters_arr[-1]) + "/";
log_path       = output_path + 'log/';
eps_path       = output_path + 'eps/';
data_path      = output_path + "data/";
pred_eps_path  = eps_path  + "pred-" + str(predcit_at_epoch) + "/"; 
pred_data_path = data_path + "pred-" + str(predcit_at_epoch) + "/";

# name of pretrain model, train model and predit_model,  data_path + 'train.pkl'
pretrain_model = "";
train_model = data_path + 'train.pkl';
if predcit_at_epoch >= train_epochs:
    predit_model = data_path + 'train.pkl';
else:
    predit_model = data_path + 'train-' + str(predcit_at_epoch) + '.pkl';

# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# mkdir and save programa
# mkdir and save programa
P.mkdir(output_path);
P.mkdir(output_path + "save-prog/");
P.cp_r("*.py", output_path + "save-prog/");
P.cp_r("func", output_path + "save-prog/");
P.cp_r("../func", output_path + "save-prog/");
P.cp_r("../../func", output_path + "save-prog/");
P.cp_r("run*", output_path + "save-prog/");


P.mkdir(log_path); P.mkdir(eps_path);  P.mkdir(data_path);
P.mkdir(data_path + "loss/");  P.mkdir(data_path + "train/");  P.mkdir(data_path + "train-pred/");  P.mkdir(data_path + "dem/");   P.mkdir(data_path + "ini/");   P.mkdir(data_path + "admm/");   P.mkdir(data_path + "network/");   P.mkdir(data_path + "middle/");

P.mkdir(eps_path + "loss/");   P.mkdir(eps_path  + "train/");  P.mkdir(eps_path  + "train-pred/");   P.mkdir(eps_path + "dem/");  P.mkdir(eps_path + "ini/");  P.mkdir(eps_path + "admm/");  P.mkdir(eps_path + "network/");  P.mkdir(eps_path + "middle/");

P.mkdir( pred_eps_path );
P.mkdir( pred_data_path );




# log.txt
P.write_txt(log_path + "log.txt",  parameters_log, type='w+');

P.write_txt(log_path + "log.txt", "x_beg is"    + str(x_beg), type='a+');
P.write_txt(log_path + "log.txt", "x_beg*dx is" + str(x_beg*dx), type='a+');
P.write_txt(log_path + "log.txt", "x_end is"    + str(x_end), type='a+');
P.write_txt(log_path + "log.txt", "x_end*dx is" + str(x_end*dx), type='a+');

P.write_txt(log_path + "log.txt", "z_beg is"    + str(z_beg), type='a+');
P.write_txt(log_path + "log.txt", "z_beg*dz is" + str(z_beg*dz), type='a+');
P.write_txt(log_path + "log.txt", "z_end is"    + str(z_end), type='a+');
P.write_txt(log_path + "log.txt", "z_end*dz is" + str(z_end*dz), type='a+');


P.write_txt(log_path + "log.txt", "home_path="+home_path, type='a+');
P.write_txt(log_path + "log.txt", "ref_path="+ref_path, type='a+');
P.write_txt(log_path + "log.txt", "psf_path="+psf_path[0], type='a+');
P.write_txt(log_path + "log.txt", "output_path="+output_path, type='a+');
P.write_txt(log_path + "log.txt", "log_path="+log_path, type='a+');
P.write_txt(log_path + "log.txt", "data_path="+data_path, type='a+');
P.write_txt(log_path + "log.txt", "eps_path="+eps_path, type='a+');


P.write_txt(log_path + "log.txt", "nx="+str(nx), type='a+');
P.write_txt(log_path + "log.txt", "nz="+str(nz), type='a+');
P.write_txt(log_path + "log.txt", "wx="+str(wx), type='a+');
P.write_txt(log_path + "log.txt", "wz="+str(wz), type='a+');
# P.write_txt(log_path + "log.txt", "ny="+str(ny), type='a+');
# P.write_txt(log_path + "log.txt", "wy="+str(wy), type='a+');
P.write_txt(log_path + "log.txt", "x_beg="+str(x_beg), type='a+');
P.write_txt(log_path + "log.txt", "x_end="+str(x_end), type='a+');
P.write_txt(log_path + "log.txt", "z_beg="+str(z_beg), type='a+');
P.write_txt(log_path + "log.txt", "z_end="+str(z_end), type='a+');
P.write_txt(log_path + "log.txt", "======================", type='a+');
P.write_txt(log_path + "log.txt", "======================", type='a+');
P.write_txt(log_path + "log.txt", "======================", type='a+');

print("check_point_bool=", check_point_bool, type(check_point_bool) );
print("network_bool=", network_bool, type(network_bool));
print("Initial_way_bool=", Initial_way_bool, type(Initial_way_bool));
print("bias_bool1=", bias_bool1, type(bias_bool1));
print("bias_bool2=", bias_bool2, type(bias_bool2));
print("admm_bool=", admm_bool, type(admm_bool));
print("lr_bool=", lr_bool, type(lr_bool));
print("train_bool=", train_bool, type(train_bool));
print("predict_bool=", predict_bool, type(predict_bool));
print("output_bool=", output_bool, type(output_bool));
print("output_other=", output_other, type(output_other));

# nnx,nny,nnz,s_n1,s_n2,s_n3,bbbl,bbbf,bbbu,bbbr,bbbb,bbbd=C.obtain_nnx_nny_nnz_from_nxnynz_wxwywz_3D(nx,ny,nz,wx,wy,wz);
# nny=1;s_n2=1;bbbf=1;bbbb=1;
