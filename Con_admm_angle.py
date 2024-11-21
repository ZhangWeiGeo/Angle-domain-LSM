#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:18:22 2023

@author: zhangjiwei
"""
# import sys;    sys.path.append("../");    from func.DEM_func import *;

from DEM_func import *
from DEM_func_3D import *
from make_data_3D import *
from admm_func import *
from admm_func_3D import *

# f = DEM_forward(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini, network_bool= network_bool, Unrolling_way=0, output_bool= output_bool).to(device);
# f.Unrolling_way=0;
















deq=0.0;


























# scheduler for https://hasty.ai/docs/mp-wiki/scheduler/cosineannealinglr
if lr_bool == 1 :
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt,lr_max_epochs, eta_min=lr_eta_min)
if lr_bool == 2:
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,lr_max_epochs, eta_min=lr_eta_min)
if lr_bool == 3:
    lr_scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
if lr_bool == 4:
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_max_epochs, gamma=0.9)



# initial for loss
loss_arr = [];    
loss1_arr = [];                                                 loss1=torch.zeros(1);
loss2_arr = [];         loss2_value=0.0;     loss_scale2=0.0;   loss2=torch.zeros(1);
loss3_arr = [];         loss3_value=0.0;     loss_scale3=0.0;   loss3=torch.zeros(1);
loss4_arr = [];         loss4_value=0.0;     loss_scale4=0.0;   loss4=torch.zeros(1);
tv_loss_arr = [];       tv_loss_value=0.0;
l1_loss_arr = [];       l1_loss_value=0.0;
lr_arr = [];            lr_value=0.0;

##coeff for different loss
if landa_1!=0:
    loss1_coeff=landa_1;   
else:
    loss1_coeff=1.0; 
    
loss2_coeff=0.0;     loss3_coeff=0.0;       loss4_coeff=0.0;


# admm variables for TV and L1 loss function
# admm variables for TV and L1 loss function
admm_s_max=0;       f_cita_max=0;       s_soft_value=0;     y_soft_value=0;

admm_resx=torch.zeros(1);  admm_resy=torch.zeros(1);  admm_resz=torch.zeros(1); #avoid error for detach
if alpha_1 != 0:
    admm_dx     = torch.zeros_like(in_x, requires_grad=False);
    admm_dy     = torch.zeros_like(in_x, requires_grad=False);
    admm_dz     = torch.zeros_like(in_x, requires_grad=False);
    admm_ux     = torch.zeros_like(in_x, requires_grad=False);
    admm_uy     = torch.zeros_like(in_x, requires_grad=False);
    admm_uz     = torch.zeros_like(in_x, requires_grad=False);
    admm_resx   = torch.zeros_like(in_x, requires_grad=False);
    admm_resy   = torch.zeros_like(in_x, requires_grad=False);
    admm_resz   = torch.zeros_like(in_x, requires_grad=False);

admm_l1_res=torch.zeros(1);#avoid error for detach   
if beta_3 != 0:
    admm_y          = torch.zeros_like(in_x, requires_grad=False);
    admm_z          = torch.zeros_like(in_x, requires_grad=False);
    admm_l1_res     = torch.zeros_like(in_x, requires_grad=False);

net_out_dx2=torch.zeros(1);#avoid error for detach
if angle_reg_para4 != 0:
    net_out_dx2 = torch.zeros_like(in_x, requires_grad=False);


net_out   = torch.zeros_like(in_x, requires_grad=False);
loss_grad = torch.zeros_like(in_x, requires_grad=False);
if CG_bool!=0:
    pre_loss_grad = torch.zeros_like(in_x, requires_grad=False);
    loss_cg_grad  = torch.zeros_like(in_x, requires_grad=False);


# training
if train_bool:
    
    for iter_m in range(0, train_epochs):
        
        if CG_bool!=0 and inner_epochs!=1:
            pre_loss_grad = torch.zeros_like(in_x, requires_grad=False);
            loss_cg_grad  = torch.zeros_like(in_x, requires_grad=False);#if inner_epochs==1, we can also use CG
        
        for inner_m in range(0, inner_epochs):
            
            if iter_m == 0 and inner_m == 0:
                # net_out2, x, Ax, Ax_y, net_x, res2 = f_train(net_out) ; # gradient descent with the step-length
                loss_grad  =  Compute_L2_gradient_with_torch_operator_3D([IDLSM_adj_op, ] , [-1.0 * ou_y, ], [1, ]);
                alpha      =  Compute_L2_step_length_with_torch_operator_3D(loss_grad, [IDLSM_for_op, ] , [-1.0 * ou_y, ], [1, ]);
                net_out    = -1.0 * alpha *loss_grad;
                net_out.requires_grad = True;
            
            
            ## just used for log, I prefer use net_out.detach(): detach the result in graph
            l1_loss         = torch.sum( torch.abs( net_out.detach() ) );
            l1_loss_value   = l1_loss .item();
            
            tv_loss         = firstorder_TV_loss_torch_3D(op_dx_torch, op_dy_torch, op_dz_torch, net_out.detach(), tv_type=tv_type) ; 
            tv_loss_value   = tv_loss.item();
            
            
            ## apply the forward modeling operator for net_out
            net_out_sys = apply_one_torch_operator_3D(IDLSM_for_op, net_out);

            ## compute residual and compute loss function for data-matching item loss1
            net_out_sys_res = net_out_sys - ou_y ;
            loss1           = 0.5 * torch.sum( net_out_sys_res * net_out_sys_res ) ;
            loss1_value     = loss1.item() ;
            


            ## compute tv misift function for loss2
            if alpha_1 != 0:
                if admm_bool==0:
                    if TV_2D_3D_bool==1:    
                        loss2 = firstorder_TV_loss_torch_3D_yz(op_dx_torch, op_dy_torch, op_dz_torch, net_out, tv_type=tv_type) ;
                    else:
                        loss2 = firstorder_TV_loss_torch_3D(op_dx_torch, op_dy_torch, op_dz_torch, net_out, tv_type=tv_type) ;
                        
                        
                else:
                    if TV_2D_3D_bool==1:
                        loss2, admm_resy, admm_resz = firstorder_admmTV_loss_torch_3D_yz(op_dx_torch, op_dy_torch, op_dz_torch, net_out, admm_dx, admm_dy, admm_dz, admm_ux, admm_uy, admm_uz);
                    else:
                        loss2, admm_resx, admm_resy, admm_resz = firstorder_admmTV_loss_torch_3D(op_dx_torch, op_dy_torch, op_dz_torch, net_out, admm_dx, admm_dy, admm_dz, admm_ux, admm_uy, admm_uz);
                    
                loss2_value = loss2.item(); #the gradient is loss2/net_out contain -1.0
                   
                if iter_m == 0 and inner_m == 0:
                    if TV_2D_3D_bool==1:    
                        admm_s_max   = cal_admm_s_max_3D_yz(op_dx_torch, op_dy_torch, op_dz_torch, net_out, tv_type=tv_type) ;
                    else:
                        admm_s_max   = cal_admm_s_max_3D(op_dx_torch, op_dy_torch, op_dz_torch, net_out, tv_type=tv_type) ;
                        
                    s_soft_value = 1.0 * alpha_1 * admm_s_max; 
                    
                if reg_change or ( iter_m == 0 and inner_m == 0 ):
                    loss_scale2 = 1.0 *  loss1_value / loss2_value ; 
                    
                loss2_coeff = ephsion_1 * loss_scale2;
                #if reg_change==true, loss2_coeff loss_scale2 is changed during the iteration

  

            ## compute l1 misift function for loss3
            if beta_3 != 0:
                if admm_bool==0:
                    loss3       = torch.sum( torch.abs(net_out) );
                else:
                    admm_l1_res = admm_y - net_out + admm_z ;
                    loss3       = 0.5 * torch.sum( admm_l1_res * admm_l1_res ) ;#the gradient is loss3/net_out =  -1.0*admm_l1_res, noted that we have -1
                
                loss3_value = loss3.item();
                
                if iter_m == 0 and inner_m == 0:
                    f_cita_max   = torch.max( torch.abs( net_out.detach() ) );
                    y_soft_value = 1.0 * beta_3 * f_cita_max ;
                
                if reg_change or ( iter_m == 0 and inner_m == 0 ):
                    loss_scale3 = 1.0 *  loss1_value / loss3_value ; 
                
                loss3_coeff =  ephsion_3 * loss_scale3 ;
                #if reg_change==true, loss3_coeff loss_scale3 is changed during the iteration
            
            
            
            ##compute loss4 for the x direction using the l2 norm of 2-order derivative
            if angle_reg_para4 != 0:
                net_out_dx2     = apply_one_torch_operator_3D(op_dx2_torch, net_out);
                loss4           = 0.5 * torch.sum(net_out_dx2 * net_out_dx2);
                
                loss4_value     = loss4.item();
                
                if reg_change or ( iter_m == 0 and inner_m == 0 ):
                    loss_scale4 = 1.0 *  loss1_value / loss4_value ;
                
                loss4_coeff     =  angle_reg_para4 * loss_scale4 ;
                #if reg_change==true, loss4_coeff loss_scale4 is changed during the iteration
            
            
            
            # cal the final loss function
            loss = 1.0 * loss1_coeff * loss1;
            if loss2_coeff!=0:
                loss =  loss + 1.0 * loss2_coeff * loss2;
            if loss3_coeff!=0:
                loss =  loss + 1.0 * loss3_coeff * loss3;
            if loss4_coeff!=0:
                loss =  loss + 1.0 * loss4_coeff * loss4;
                
            loss_value = loss.item();
            
            ## hook gradient
            ## some operations
            
            ## hook gradient
            
            
            # backward for computing gradient and optimize  and optimize
            loss_grad        =  torch.zeros_like(in_x, requires_grad=False); # set zeros, must be set zeros,
            loss_grad        =  autograd.grad(loss, net_out, retain_graph=True)[0];
            
            A_list     = [IDLSM_adj_op, op_dx_torch_H, op_dy_torch_H, op_dz_torch_H, "I", op_dx2_torch_H];
            res_list   = [net_out_sys_res.detach(), -1.0 * admm_resx.detach(), -1.0 * admm_resy.detach(), -1.0 * admm_resz.detach(), -1.0 * admm_l1_res.detach(), net_out_dx2.detach()]; ###noted, we must multiply -1 for admm residual.
            scale_list = [loss1_coeff, loss2_coeff, loss2_coeff, loss2_coeff, loss3_coeff, loss4_coeff];
            
            if iter_m <=3 and inner_m == 0:
                check_loss_grad  =   Compute_L2_gradient_with_torch_operator_3D(A_list, res_list, scale_list);
                check_grad_bool  =   torch.allclose(loss_grad, check_loss_grad, atol=1e-03);

            
            if CG_bool !=0:
                loss_cg_grad, beta  = Compute_CG_direction(loss_cg_grad, loss_grad, pre_loss_grad);
                pre_loss_grad[:]    =  1.0*loss_grad[:]; #save previous gradient
                
                loss_grad[:]        =  1.0*loss_cg_grad[:]; ###now step length is negative value 
                file='beta={}\n'.format(beta );print(file); P.write_txt(log_path+"log.txt", file, type='a+');
            
            A_list     = [IDLSM_for_op, op_dx_torch, op_dy_torch, op_dz_torch, "I", op_dx2_torch]
            alpha      =   Compute_L2_step_length_with_torch_operator_3D(loss_cg_grad.detach(), A_list, res_list, scale_list);#derived from m = m -ak*dk
        
            ##noted there is -=
            net_out         = net_out - alpha * loss_grad;	lr_value= alpha.item();
            # net_out         = net_out + alpha * loss_grad;	lr_value= alpha.item();#now we revise this as +
					

            loss_arr.append   ( loss_value);
            loss1_arr.append  ( loss1_value);
            loss2_arr.append  ( loss2_value);
            loss3_arr.append  ( loss3_value);
            loss4_arr.append  ( loss4_value);
            tv_loss_arr.append( tv_loss_value);
            l1_loss_arr.append( l1_loss_value);
            lr_arr.append(lr_value);
            
            if np.isnan(float(loss.item())):
                file='loss is nan while testing\n'
                print( file ); P.write_txt(log_path+"log.txt", file + file + file + file, type='a+') ;
                raise ValueError(file)
                
            #log.txt
            output_log1(iter_m, loss_value, loss1_value, loss2_value, loss3_value, tv_loss_value, l1_loss_value, loss_scale2, loss_scale3, loss1_coeff, loss2_coeff, loss3_coeff, lr_value, admm_s_max, f_cita_max, s_soft_value, y_soft_value, deq, DEM_or_DU, log_path, eps_dpi=eps_dpi) ;
            file='iteration={} loss4[{}]={}\n'.format(iter_m, iter_m, loss4_value);
            print(file); P.write_txt(log_path+"log.txt", file, type='a+');
            file='iteration={} loss_scale4[{}]={}\n'.format(iter_m, iter_m, loss_scale4);
            print(file); P.write_txt(log_path+"log.txt", file, type='a+');
            file='iteration={} loss4_coeff[{}]={}\n'.format(iter_m, iter_m, loss4_coeff);
            print(file); P.write_txt(log_path+"log.txt", file, type='a+');
        
        
        # for iter_m in range(0, train_epochs):   
        #update admm primal(aux) and dual variables, we need to update admm variables in out of inner_epochs
        if alpha_1 != 0 and admm_bool != 0:
            admm_update_for_tv_3D(op_dx_torch, op_dy_torch, op_dz_torch, net_out, admm_dx, admm_dy, admm_dz, admm_ux, admm_uy, admm_uz, s_soft_value, tv_type=tv_type) ;
        if beta_3  != 0 and admm_bool != 0:
            admm_update_for_l1_3D(op_dx_torch, op_dy_torch, op_dz_torch, net_out, admm_y, admm_z, y_soft_value) ;

        # for iter_m in range(0, train_epochs):   
        
            
        file='check_grad_bool {}\n'.format(check_grad_bool); print(file);P.write_txt(log_path+"log.txt",file,type='a+');
        
        
        
        
        if (iter_m) % output_epochs == 0 or iter_m == (train_epochs-1):
            output_loss_eps(iter_m, loss1_arr, loss2_arr, loss3_arr, tv_loss_arr, l1_loss_arr, lr_arr, data_path + "admm/", eps_path + "admm/", log_path, eps_dpi=eps_dpi) ;   
            if (net_out.shape[1]!=1):
                output_train_3D_eps(iter_m, deq, data_path + "admm/", eps_path + "admm/", log_path, inv_scale * forward_scale* net_out_sys, inv_scale * forward_scale* net_out_sys_res, inv_scale * net_out, inv_scale * forward_scale*ou_y, ref_arr, eps_dpi=eps_dpi) ;  
            else:
                output_train_eps(iter_m, deq, data_path + "admm/", eps_path + "admm/", log_path, net_out_sys, net_out_sys_res, net_out, ou_y, inv_scale, forward_scale, ref_arr, eps_dpi=eps_dpi) ; 

#os.system("python Con_admm_angle_compare.py");
