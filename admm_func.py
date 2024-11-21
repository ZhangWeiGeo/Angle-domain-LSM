#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:02:22 2023

@author: zhangw0c
"""

import torch  # from Library import *
import sys

def apply_one_torch_operator(For_ward, net_out):
    
    output_arr = torch.zeros_like(net_out, requires_grad=False, device=net_out.device) ;

    size1, size2, _, _ = net_out.shape ;
    
    for ix in range(0, size1):
        for iz in range(0, size2):
            x = net_out[ix,iz,:,:]
            Ax = For_ward.apply(x);
            output_arr[ix,iz,:,:] = 1.0*Ax;
    return output_arr

def cal_admm_s_max(Lx, Lz, net_out, tv_type=0):
    
    size1, size2, _, _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    output = net_out.detach()
    
    for ix in range(0,size1):
        for iz in range(0,size2):  
            in_data = output[ix,iz,:,:]
            Lx_in = Lx.apply(in_data);
            Lz_in = Lz.apply(in_data);
            
    return torch.max( torch.sqrt( Lx_in * Lx_in + Lz_in * Lz_in ) );

def firstorder_TV_loss_torch(Lx, Lz, net_out, tv_type=0):
    
    size1, size2, _, _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    
    for ix in range(0,size1):
        for iz in range(0,size2):
            
            in_data = net_out[ix,iz,:,:]
            Lx_in = Lx.apply(in_data);
            Lz_in = Lz.apply(in_data);
            
            if tv_type ==0:
                loss = loss + torch.sum ( torch.sqrt( Lx_in * Lx_in + Lz_in * Lz_in ) ) ;
            elif tv_type ==1:
                loss = loss + torch.sum ( torch.abs(Lx_in) + torch.abs(Lz_in) ) ;
            else:
                loss = loss + torch.sum ( Lx_in * Lx_in + Lz_in * Lz_in ) ;

    return 0.5*loss

def firstorder_admmTV_loss_torch(Lx, Lz, net_out, admm_dx, admm_dz, admm_ux, admm_uz):
    
    size1, size2, _, _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    
    for ix in range(0,size1):
        for iz in range(0,size2):
            
            in_data =  net_out[ix,iz,:,:]
            Lx_in   =  Lx.apply( in_data ); 
            Lz_in   =  Lz.apply( in_data );
            resx    =  admm_dx[ix,iz,:,:] - Lx_in + admm_ux;
            resz    =  admm_dz[ix,iz,:,:] - Lz_in + admm_uz;
            loss    = loss + 0.5 * torch.sum( resx * resx + resz * resz );

    return loss, resx, resz

def admm_update_for_tv(op_dx_torch, op_dz_torch, net_out, admm_dx, admm_dz, admm_ux, admm_uz, s_soft_value, tv_type=0):
    
    size1, size2, _, _ = net_out.shape; 
    
    m_f = net_out.detach() ;    ##otherwise, it does not free memory
    
    gama = 1.0 * s_soft_value
    
    for ix in range(0,size1):
        for iz in range(0,size2):
            in_data = m_f[ix,iz,:,:]
            Lx_in   = op_dx_torch.apply(in_data);
            Lz_in   = op_dz_torch.apply(in_data);
    
            resx    = Lx_in - admm_ux[ix,iz,:,:];
            resz    = Lz_in - admm_uz[ix,iz,:,:];
            
            ##update dx, dy, dz
            if tv_type==0:
                sk    = torch.sqrt( resx * resx + resz * resz );
                scale = sk - gama;
                scale[scale<0] = 0.0;
                sk[sk == 0] = 1e-13; ##used for stablized
                admm_dx[ix,iz,:,:] = 1.0 * scale * resx / sk;
                admm_dz[ix,iz,:,:] = 1.0 * scale * resz / sk;
                
            else:
                tmp1 = torch.sign( resx );
                tmp2 = torch.abs( resx)  -gama
                tmp2[tmp2<0] = 0.0;
                admm_dx[ix,iz,:,:] = 1.0 * tmp1 * tmp2;
                
                tmp1 = torch.sign( resz );
                tmp2 = torch.abs( resz) -gama
                tmp2[tmp2<0] = 0.0;
                admm_dz[ix,iz,:,:] = 1.0 * tmp1 * tmp2;
                
            ##update ux, uy, and uz
            admm_ux[ix,iz,:,:] = admm_ux[ix,iz,:,:] + ( admm_dx[ix,iz,:,:] - Lx_in );
            admm_uz[ix,iz,:,:] = admm_uz[ix,iz,:,:] + ( admm_dz[ix,iz,:,:] - Lz_in );

def admm_update_for_l1(op_dx_torch, op_dz_torch, net_out, admm_y, admm_z, y_soft_value):
    
    size1, size2, _, _ = net_out.shape ;
    
    m_f = net_out.detach() ;  ##otherwise, it does not free memory
    
    gama = 1.0 * y_soft_value ;
    
    for ix in range(0,size1):
        for iz in range(0,size2):
            
            res  = m_f[ix,iz,:,:] - admm_z[ix,iz,:,:];
            tmp1 = torch.sign( res );
            tmp2 = torch.abs( res)  - gama
            tmp2[tmp2<0] = 0.0;
            admm_y[ix,iz,:,:] = 1.0 * tmp1 * tmp2;
            
            admm_z[ix,iz,:,:] = admm_z[ix,iz,:,:] + ( admm_y[ix,iz,:,:] - m_f[ix,iz,:,:] );


def Compute_L2_gradient_with_torch_operator_2D(A_list, res_list, scale_list):
    
    if len(A_list)!=len(res_list) or len(A_list)!=len(scale_list):
        print("len(A_list) is ", len(A_list));print("len(res_list) is ", len(res_list));print("len(scale_list) is ", len(scale_list));
        sys.exit("We must input the same lenth for A_list, res_list, scale_list");
        
    output  = torch.zeros_like(res_list[0], requires_grad=False);
    
    for i in range(0, len(A_list)):
        scale = scale_list[i]
        if scale!=0:
            A     = A_list[i];
            res   = res_list[i]; #
            
            if A!="I":
                A_res  = apply_one_torch_operator(A, res);
            else:
                A_res  = 1.0*res;
        
            output = output + 1.0*scale*A_res;
    
    return output

def Compute_L2_gradient_with_torch_operator(A1, res1, scale1, A2=0, res2=0, scale2=0, A3=0, res3=0, scale3=0, A4=0, res4=0, scale4=0):
    
    A1_g  = torch.zeros_like(res1, requires_grad=False, device=res1.device);
    A2_g  = torch.zeros_like(res1, requires_grad=False, device=res1.device);
    A3_g  = torch.zeros_like(res1, requires_grad=False, device=res1.device);
    A4_g  = torch.zeros_like(res1, requires_grad=False, device=res1.device);

    if scale1!=0:
        A1_g = apply_one_torch_operator(A1, res1);
    
    if scale2!=0:
        A2_g = apply_one_torch_operator(A2, res2);
    
    if scale3!=0:
        A3_g = apply_one_torch_operator(A3, res3);
    
    if scale4!=0:
        A4_g = 1.0*res4;
        
    return  scale1*A1_g +  scale2*A2_g + scale3*A3_g + scale4*A4_g 
        
def Compute_L2_step_length_with_torch_operator(loss_grad, A1, res1, scale1, A2=0, res2=0, scale2=0, A3=0, res3=0, scale3=0, A4=0, res4=0, scale4=0):
    
    alpha = torch.zeros(1, requires_grad=False, device=res1.device);
    A1_g  = torch.zeros_like(res1, requires_grad=False, device=res1.device);
    A2_g  = torch.zeros_like(res1, requires_grad=False, device=res1.device);
    A3_g  = torch.zeros_like(res1, requires_grad=False, device=res1.device);
    A4_g  = torch.zeros_like(res1, requires_grad=False, device=res1.device);
    
    if scale1!=0:
        A1_g = apply_one_torch_operator(A1, loss_grad);
    
    if scale2!=0:
        A2_g = apply_one_torch_operator(A2, loss_grad);
    
    if scale3!=0:
        A3_g = apply_one_torch_operator(A3, loss_grad);
    
    if scale4!=0:
        A4_g = 1.0*loss_grad;
    
    
    b = scale1 * torch.sum(torch.multiply(A1_g, A1_g)) + scale2 * torch.sum(torch.multiply(A2_g, A2_g)) + scale3 * torch.sum(torch.multiply(A3_g, A3_g)) + scale4 * torch.sum(torch.multiply(A4_g, A4_g))
    a = scale1 * torch.sum(torch.multiply(A1_g, res1)) + scale2 * torch.sum(torch.multiply(A2_g, res2)) + scale3 * torch.sum(torch.multiply(A3_g, res3)) + scale4 * torch.sum(torch.multiply(A4_g, res4));
    
    alpha = 1.0*a/b;
    
    return alpha 

def Compute_L2_step_length_with_torch_operator_2D(loss_grad, A_list, res_list, scale_list):
    #derived from m = m -ak*dk
    if len(A_list)!=len(res_list) or len(A_list)!=len(scale_list):
        print("len(A_list) is ", len(A_list));print("len(res_list) is ", len(res_list));print("len(scale_list) is ", len(scale_list));
        sys.exit("We must input the same lenth for A_list, res_list, scale_list");
         
    alpha = torch.zeros(1, requires_grad=False, device=res_list[0].device);
    a = torch.zeros(1, requires_grad=False, device=res_list[0].device);
    b = torch.zeros(1, requires_grad=False, device=res_list[0].device);
        
    for i in range(0, len(A_list)):
        scale = scale_list[i]
        if scale!=0:
            A     = A_list[i];
            res   = res_list[i]; #.detach(); #we move the detach to here, this is not correct
            
            if A!="I":
                A_g   = apply_one_torch_operator(A, loss_grad);
            else:
                A_g   = 1.0*loss_grad;
            
            b = b + scale * torch.sum(torch.multiply(A_g, A_g));
            a = a + scale * torch.sum(torch.multiply(A_g, res));
    
    alpha = 1.0*a/b;
    
    return alpha 

def Compute_cgls_step(cg_grad, loss_grad, A_list, scale_list):
    
    if len(A_list)!=len(scale_list):
        print("len(A_list) is ", len(A_list));print("len(scale_list) is ", len(scale_list));
        sys.exit("We must input the same lenth for A_list, res_list, scale_list");
         
    alpha = torch.zeros(1, requires_grad=False, device=loss_grad[0].device);
    a = torch.zeros(1, requires_grad=False, device=loss_grad[0].device);
    b = torch.zeros(1, requires_grad=False, device=loss_grad[0].device);
    
    a = torch.sum(torch.multiply( loss_grad, loss_grad) );
    
    for i in range(0, len(A_list)):
        scale = scale_list[i]
        if scale!=0:
            A     = A_list[i];
            
            if A!="I":
                A_g   = apply_one_torch_operator(A, cg_grad);
            else:
                A_g   = 1.0*cg_grad;
            
            b = b + scale * torch.sum(torch.multiply(A_g, A_g));
             
    alpha = 1.0*a/b;
    
    return alpha 


def Compute_cgls_beta(cg_grad, inv_grad, pre_grad):
    
    beta1 = torch.sum( torch.multiply(inv_grad, inv_grad) ) 
    beta2 = torch.sum( torch.multiply(pre_grad, pre_grad) ) 
    
    if beta2!=0:
        beta  = beta1 / beta2
    else:
        beta  = 0;
        
    return beta*cg_grad + inv_grad, beta


# loss_grad, beta = Compute_CG_direction(loss_cg_grad.detach(), loss_grad.detach(), pre_loss_grad.detach());
def Compute_CG_direction(cg, g1, g0):
    
    beta_hs1 = torch.sum(torch.multiply(g1, g1-g0) )
    beta_hs2 = torch.sum(torch.multiply(cg, g1-g0) )
    if beta_hs2 !=0:
        beta_hs = beta_hs1 / beta_hs2;
    else:
        beta_hs = 0;
            
    beta_dy1 = torch.sum(torch.multiply(g1, g1) )
        
    if beta_hs2 !=0:
        beta_dy = beta_dy1 / beta_hs2;
    else:
        beta_dy = 0;
        
    beta = max(min(beta_hs, beta_dy), 0);
    
    return -1.0*g1[:] + beta*cg[:], beta
    
    
# class DEM_forward(nn.Module):
#     def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.2, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001, network_bool=3, Unrolling_way=1, output_bool=False):
#         super().__init__()
#         print("class DEM_forward(nn.Module)");
#         print("in_channels=",in_channels);
#         print("ou_channels=",ou_channels);
#         print("filter_num=",filter_num);
#         print("kernel_size=",kernel_size);
#         print("padding=",padding);
#         print("padding_mode=",padding_mode);
#         print("is_deconv=",is_deconv);
#         print("is_batchnorm=",is_batchnorm);
#         print("spectral_norm=",spectral_norm);
        
#         print("init_weight=",init_weight);
#         print("bias_bool1=",bias_bool1);
#         print("bias_bool2=",bias_bool2);
#         print("Leaky=",Leaky);
#         print("neu_in_num=",neu_in_num);
#         print("neu_ou_num=",neu_ou_num);
#         print("drop_layer=",drop_layer);
#         print("drop_out=",drop_out);
#         print("neu_ini=",neu_ini);
        
#         # self.res_norm = []
#         self.Unrolling_way = Unrolling_way;
#         self.network_bool  = network_bool;
#         self.drop_layer    = drop_layer;
#         self.drop_out      = drop_out;
        
#         if self.network_bool == 4:
#             self.net = N_unet.UnetModel(in_channels, ou_channels, filter_num, 4, drop_out);
        
#         if self.network_bool == 5:
#             self.net = N_unet.DnCNN(in_channels);
            
            
#         elif self.network_bool == 1:
#             self.net = N.DEM_net_1(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);#U-net
        
#         elif self.network_bool == 2:
#             self.net = N.DEM_net_2(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);#Wu2019
        
#         elif self.network_bool == 3:
#             self.net = N.DEM_net_3(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);#Standard Conv with Batch normal
            
#         elif self.network_bool == 30:
#             self.net = N.DEM_net_30(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
            
#         elif self.network_bool == 33:
#             self.net = N.DEM_net_33(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
            
#         elif self.network_bool == 31:
#             self.net = N.DEM_net_31(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);    
        
#         elif self.network_bool == 32:
#             self.net = N.DEM_net_32(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
          
            
#         elif self.network_bool == 40:
#             self.net = N.DEM_net_40(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
#         elif self.network_bool == 41:
#             self.net = N.DEM_net_41(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
       
            
#         elif self.network_bool == 50:
#             self.net = N.DEM_net_50(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
#         elif self.network_bool == 51:
#             self.net = N.DEM_net_51(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
            
            
#         elif self.network_bool == 300:
#             self.net = N.DEM_net_300(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
        
#         if self.network_bool !=0:
#             # Initialization of Parameters        
#             # N.Initialization_net(self.net, init_weight, output_txt=False);
#             self.net = self.net.to(device);
        
#     def forward(self, yy, xx, For_ward, Adj_joint, check_point_bool=True, net_iter_input_nor=True):
#         #yy is obs, xx is the model parameters, for a inverse problem    
#         size1, size2, nx, nz = xx.shape;#print("xx.requires_grad",xx.requires_grad)
#         output_tensor = torch.zeros_like(xx,requires_grad=False);# print("output_tensor.requires_grad",output_tensor.requires_grad)
            
#         for ix in range(0,size1):
#             for iz in range(0,size2):
                
#                 x = xx[ix,iz,:,:];
#                 y = yy[ix,iz,:,:];
            
#                 Ax = For_ward.apply(x);
#                 Ax_y = Ax - y;
                
#                 gk = Adj_joint.apply(Ax_y);
#                 Agk = For_ward.apply(gk);
#                 b = torch.sum(torch.multiply(Agk,Agk));
#                 a = torch.sum(torch.multiply(Agk,Ax_y));
#                 alpha = 1.0*a/b;
#                 #print("alpha is {}".format(alpha));print("alpha.device is {}".format(alpha.device));print("alpha.dtype is {}".format(alpha.dtype));
            
#                 x = x - alpha*gk;
                
#                 # #why it does not free memeroy?
#                 # if self.output_bool: 
#                 #     self.res_norm = self.res_norm + ((torch.sum(torch.multiply(Ax_y,Ax_y))));
            
#                 input_tensor = 1.0*x;
                
#                 output_tensor[ix,iz,:,:] = 1.0*input_tensor;

#         if net_iter_input_nor:
#             max_abs_val = torch.max( torch.abs(xx) );
#             if max_abs_val!=0:
#                 xx    = xx/max_abs_val; # only this function works, it does not work out of this function for the revision of xx

#         if self.Unrolling_way==0:
            
#             return output_tensor, x, Ax, Ax_y, x, torch.sum(torch.multiply(Ax_y,Ax_y))

#         elif self.Unrolling_way==1:
            
#             if check_point_bool:
#                 net_x = alpha * checkpoint( self.net, xx, use_reentrant=False);
#             else:
#                 net_x = alpha * self.net(xx)
#             output = output_tensor - net_x
            
#             return output, x, Ax, Ax_y, net_x, torch.sum(torch.multiply(Ax_y,Ax_y))
 
#         elif self.Unrolling_way==2:
            
#             if check_point_bool:
#                 output = checkpoint( self.net, output_tensor, use_reentrant=False);
#             else:
#                 output=self.net(output_tensor)
                
#             # print("output.requires_grad",output.requires_grad)
#             return output, x, Ax, Ax_y, x, torch.sum(torch.multiply(Ax_y,Ax_y))