#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:34:37 2023

@author: zhangw0c
"""

import torch
import sys


def apply_one_torch_operator_list(For_ward_list, net_out):
    
    output_arr = torch.zeros_like(net_out, requires_grad=False) ;

    size1, size2, _, _ = net_out.shape ;

    for iangle, For_ward in enumerate(For_ward_list):

        for ix in range(0, size1):
            # for iz in range(0, size2):
                x = net_out[ix, iangle, :, :]
                Ax = For_ward.apply(x);
                output_arr[ix, iangle, :, :] = 1.0*Ax;
    return output_arr

def apply_one_torch_operator_3D_list(For_ward_list, net_out):
    
    output_arr = torch.zeros_like(net_out, requires_grad=False) ;

    size1, size2, _, _ = net_out.shape ;

    for iangle, For_ward in enumerate(For_ward_list):

        for ix in range(0, size1):
            # for iz in range(0, size2):
                x = net_out[ix, iangle, :, :]
                Ax = For_ward.apply(x);
                output_arr[ix, iangle, :, :] = 1.0*Ax;
    return output_arr


def apply_one_torch_operator_3D(For_ward, net_out):
    
    output_arr = torch.zeros_like(net_out, requires_grad=False) ;

    size1, size2, _, _ = net_out.shape ;
    
    for ix in range(0, size1):
        # for iz in range(0, size2):
            x = net_out[ix,:,:,:]
            Ax = For_ward.apply(x);
            output_arr[ix,:,:,:] = 1.0*Ax;
    return output_arr





def cal_admm_s_max_3D(Lx, Ly, Lz, net_out, tv_type=0):
    
    size1, size2, _ , _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    output = net_out.detach()
    
    for ix in range(0,size1):
        # for iz in range(0,size2):  
            in_data = output[ix,:,:,:]
            Lx_in = Lx.apply(in_data);
            Ly_in = Ly.apply(in_data);
            Lz_in = Lz.apply(in_data);
            
    return torch.max( torch.sqrt( Lx_in * Lx_in + Ly_in * Ly_in + Lz_in * Lz_in ) );

def cal_admm_s_max_3D_yz(Lx, Ly, Lz, net_out, tv_type=0):
    
    size1, size2, _ , _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    output = net_out.detach()
    
    for ix in range(0,size1):
        # for iz in range(0,size2):  
            in_data = output[ix,:,:,:]
            # Lx_in = Lx.apply(in_data);
            Ly_in = Ly.apply(in_data);
            Lz_in = Lz.apply(in_data);
            
    return torch.max( torch.sqrt( Ly_in * Ly_in + Lz_in * Lz_in ) );




def firstorder_TV_loss_torch_3D(Lx, Ly, Lz, net_out, tv_type=0):
    
    size1,  size2, _, _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    
    for ix in range(0,size1):
        # for iz in range(0,size2):
            
            in_data = net_out[ix,:,:,:]
            Lx_in = Lx.apply(in_data);
            Ly_in = Ly.apply(in_data);
            Lz_in = Lz.apply(in_data);
            
            if tv_type ==0:
                loss = loss + torch.sum ( torch.sqrt( Lx_in * Lx_in + Ly_in * Ly_in + Lz_in * Lz_in ) ) ;
            elif tv_type ==1:
                loss = loss + torch.sum ( torch.abs(Lx_in) + torch.abs(Ly_in) + torch.abs(Lz_in) ) ;
            else:
                loss = loss + torch.sum ( Lx_in * Lx_in + Ly_in * Ly_in + Lz_in * Lz_in ) ;

    return 0.5*loss

def firstorder_TV_loss_torch_3D_yz(Lx, Ly, Lz, net_out, tv_type=0):
    
    size1,  size2, _, _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    
    for ix in range(0,size1):
        # for iz in range(0,size2):
            
            in_data = net_out[ix,:,:,:]
            # Lx_in = Lx.apply(in_data);
            Ly_in = Ly.apply(in_data);
            Lz_in = Lz.apply(in_data);
            
            if tv_type ==0:
                loss = loss + torch.sum ( Ly_in * Ly_in + Lz_in * Lz_in );
            elif tv_type ==1:
                loss = loss + torch.sum ( torch.abs(Ly_in) + torch.abs(Lz_in) ) ;
            else:
                loss = loss + torch.sum ( Ly_in * Ly_in + Lz_in * Lz_in ) ;

    return 0.5*loss


def firstorder_admmTV_loss_torch_3D(Lx, Ly, Lz, net_out, admm_dx, admm_dy, admm_dz, admm_ux, admm_uy, admm_uz):
    
    size1, size2, _, _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    
    for ix in range(0,size1):
        # for iz in range(0,size2):
            
            in_data =  net_out[ix,:,:,:]
            Lx_in   =  Lx.apply( in_data ); 
            Ly_in   =  Ly.apply( in_data );
            Lz_in   =  Lz.apply( in_data );
            resx    =  admm_dx[ix,:,:,:] - Lx_in + admm_ux;
            resy    =  admm_dy[ix,:,:,:] - Ly_in + admm_uy;
            resz    =  admm_dz[ix,:,:,:] - Lz_in + admm_uz;
            loss    = loss + 0.5 * torch.sum( resx * resx + resy * resy + resz * resz );

    return loss, resx, resy, resz

def firstorder_admmTV_loss_torch_3D_yz(Lx, Ly, Lz, net_out, admm_dx, admm_dy, admm_dz, admm_ux, admm_uy, admm_uz):
    
    size1, size2, _, _ = net_out.shape;
    loss = torch.zeros(1, requires_grad=False, device=net_out.device);
    
    for ix in range(0,size1):
        # for iz in range(0,size2):
            
            in_data =  net_out[ix,:,:,:]
            # Lx_in   =  Lx.apply( in_data ); 
            Ly_in   =  Ly.apply( in_data );
            Lz_in   =  Lz.apply( in_data );
            # resx    =  admm_dx[ix,:,:,:] - Lx_in + admm_ux;
            resy    =  admm_dy[ix,:,:,:] - Ly_in + admm_uy;
            resz    =  admm_dz[ix,:,:,:] - Lz_in + admm_uz;
            loss    = loss + 0.5 * torch.sum( resy * resy + resz * resz );

    return loss, resy, resz




def admm_update_for_tv_3D(op_dx_torch, op_dy_torch, op_dz_torch, net_out, admm_dx, admm_dy, admm_dz, admm_ux, admm_uy, admm_uz, s_soft_value, tv_type=0):
    
    size1, size2, _, _ = net_out.shape; 
    
    m_f = net_out.detach() ;    ##otherwise, it does not free memory
    
    gama = 1.0 * s_soft_value
    
    for ix in range(0,size1):
        # for iz in range(0,size2):
            in_data = m_f[ix,:,:,:]
            Lx_in   = op_dx_torch.apply(in_data);
            Ly_in   = op_dy_torch.apply(in_data);
            Lz_in   = op_dz_torch.apply(in_data);
    
            resx    = Lx_in - admm_ux[ix,:,:,:];
            resy    = Ly_in - admm_uy[ix,:,:,:];
            resz    = Lz_in - admm_uz[ix,:,:,:];
            
            ##update dx, dy, dz
            if tv_type==0:
                sk    = torch.sqrt( resx * resx + resy * resy + resz * resz );
                scale = sk - gama;
                scale[scale<0] = 0.0;
                sk[sk == 0] = 1e-13; ##used for stablized
                admm_dx[ix,:,:,:] = 1.0 * scale * resx / sk;
                admm_dy[ix,:,:,:] = 1.0 * scale * resy / sk;
                admm_dz[ix,:,:,:] = 1.0 * scale * resz / sk;
                
            else:
                tmp1 = torch.sign( resx );
                tmp2 = torch.abs( resx)  -gama
                tmp2[tmp2<0] = 0.0;
                admm_dx[ix,:,:,:] = 1.0 * tmp1 * tmp2;
                
                tmp1 = torch.sign( resy );
                tmp2 = torch.abs( resy)  -gama
                tmp2[tmp2<0] = 0.0;
                admm_dx[ix,:,:,:] = 1.0 * tmp1 * tmp2;
                
                tmp1 = torch.sign( resz );
                tmp2 = torch.abs( resz) -gama
                tmp2[tmp2<0] = 0.0;
                admm_dz[ix,:,:,:] = 1.0 * tmp1 * tmp2;
                
            ##update ux, uy, and uz
            admm_ux[ix,:,:,:] = admm_ux[ix,:,:,:] + ( admm_dx[ix,:,:,:] - Lx_in );
            admm_uy[ix,:,:,:] = admm_uy[ix,:,:,:] + ( admm_dy[ix,:,:,:] - Ly_in );
            admm_uz[ix,:,:,:] = admm_uz[ix,:,:,:] + ( admm_dz[ix,:,:,:] - Lz_in );



def admm_update_for_l1_3D(op_dx_torch, op_dy_torch, op_dz_torch, net_out, admm_y, admm_z, y_soft_value):
    
    size1, size2, _, _ = net_out.shape ;
    
    m_f = net_out.detach() ;  ##otherwise, it does not free memory
    
    gama = 1.0 * y_soft_value ;
    
    for ix in range(0,size1):
        # for iz in range(0,size2):
            
            res  = m_f[ix,:,:,:] - admm_z[ix,:,:,:];
            tmp1 = torch.sign( res );
            tmp2 = torch.abs( res)  - gama
            tmp2[tmp2<0] = 0.0;
            admm_y[ix,:,:,:] = 1.0 * tmp1 * tmp2;
            
            admm_z[ix,:,:,:] = admm_z[ix,:,:,:] + ( admm_y[ix,:,:,:] - m_f[ix,:,:,:] );


def Compute_L2_gradient_with_torch_operator_3D_old(A1, res1, scale1, A2=0, res2=0, scale2=0, A3=0, res3=0, scale3=0, A4=0, res4=0, scale4=0, A5=0, res5=0, scale5=0, A6=0, res6=0, scale6=0):
    
    if scale1!=0:
        A1_g  = torch.zeros_like(res1, requires_grad=False);
        A1_g  = apply_one_torch_operator_3D(A1, res1);
    else:
        A1_g  = 0.0;
    
    if scale2!=0:
        A2_g  = torch.zeros_like(res1, requires_grad=False);
        A2_g = apply_one_torch_operator_3D(A2, res2);
    else:
        A2_g = 0.0;
    
    if scale3!=0:
        A3_g  = torch.zeros_like(res1, requires_grad=False);
        A3_g  = apply_one_torch_operator_3D(A3, res3);
    else:
        A3_g  = 0.0;
        
    if scale4!=0:
        A4_g  = torch.zeros_like(res1, requires_grad=False);
        A4_g = apply_one_torch_operator_3D(A4, res4);
    else:
        A4_g = 0.0;
        
    if scale5!=0:
        A5_g  = torch.zeros_like(res1, requires_grad=False);
        A5_g  = 1.0*res5;
    else:
        A5_g  = 0.0;
        
    if scale6!=0:
        A6_g  = torch.zeros_like(res1, requires_grad=False);
        A6_g  = 1.0*res6;
    else:
        A6_g  = 0.0;

    return  scale1*A1_g +  scale2*A2_g + scale3*A3_g + scale4*A4_g + scale5*A5_g  + scale6*A6_g 


def Compute_L2_gradient_with_torch_operator_3D(A_list, res_list, scale_list):
    
    if len(A_list)!=len(res_list) or len(A_list)!=len(scale_list):
        print("len(A_list) is ", len(A_list));print("len(res_list) is ", len(res_list));print("len(scale_list) is ", len(scale_list));
        sys.exit("We must input the same lenth for A_list, res_list, scale_list");
        
    output  = torch.zeros_like(res_list[0], requires_grad=False);
    
    for i in range(0, len(A_list)):
        scale = scale_list[i]
        if scale!=0:
            A     = A_list[i];
            res   = res_list[i]; #.detach() ; #we move the detach to here, this is not correct
            
            if A!="I":
                A_res  = apply_one_torch_operator_3D(A, res);
            else:
                A_res  = 1.0*res;
        
            output = output + 1.0*scale*A_res;
    
    return output

def Compute_L2_step_length_with_torch_operator_3D(loss_grad, A_list, res_list, scale_list):
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
                A_g   = apply_one_torch_operator_3D(A, loss_grad);
            else:
                A_g   = 1.0*loss_grad;
            
            b = b + scale * torch.sum(torch.multiply(A_g, A_g));
            a = a + scale * torch.sum(torch.multiply(A_g, res));
    
    alpha = 1.0*a/b;
    # alpha = -1.0*a/b;
    
    return alpha 


def Compute_L2_step_length_with_torch_operator_3D_old(loss_grad, A1, res1, scale1, A2=0, res2=0, scale2=0, A3=0, res3=0, scale3=0, A4=0, res4=0, scale4=0, A5=0, res5=0, scale5=0, A6=0, res6=0, scale6=0):
    
    alpha = torch.zeros(1, requires_grad=False, device=res1.device);
    a = torch.zeros(1, requires_grad=False, device=res1.device);
    b = torch.zeros(1, requires_grad=False, device=res1.device);

    if scale1!=0:
        A1_g  = torch.zeros_like(res1, requires_grad=False);
        A1_g  = apply_one_torch_operator_3D(A1, loss_grad);
        b = b + scale1 * torch.sum(torch.multiply(A1_g, A1_g));
        a = a + scale1 * torch.sum(torch.multiply(A1_g, res1));
    
    if scale2!=0:
        A2_g  = torch.zeros_like(res1, requires_grad=False);
        A2_g  = apply_one_torch_operator_3D(A2, loss_grad);
        b = b + scale2 * torch.sum(torch.multiply(A2_g, A2_g));
        a = a + scale2 * torch.sum(torch.multiply(A2_g, res2));
    
    if scale3!=0:
        A3_g  = torch.zeros_like(res1, requires_grad=False);
        A3_g  = apply_one_torch_operator_3D(A3, loss_grad);
        b = b + scale3 * torch.sum(torch.multiply(A3_g, A3_g));
        a = a + scale3 * torch.sum(torch.multiply(A3_g, res3));
        
    if scale4!=0:
        A4_g  = torch.zeros_like(res1, requires_grad=False);
        A4_g  = apply_one_torch_operator_3D(A4, loss_grad);
        b = b + scale4 * torch.sum(torch.multiply(A4_g, A4_g));
        a = a + scale4 * torch.sum(torch.multiply(A4_g, res4));
        
    if scale5!=0:
        A5_g  = torch.zeros_like(res1, requires_grad=False);
        A5_g  = 1.0*loss_grad;
        b = b + scale5 * torch.sum(torch.multiply(A5_g, A5_g));
        a = a + scale5 * torch.sum(torch.multiply(A5_g, res5));
    
        
    if scale6!=0:
        A6_g  = torch.zeros_like(res1, requires_grad=False);
        A6_g  = apply_one_torch_operator_3D(A6, loss_grad);
        b = b + scale6 * torch.sum(torch.multiply(A6_g, A6_g));
        a = a + scale6 * torch.sum(torch.multiply(A6_g, res6));
    
      
    # b = scale1 * torch.sum(torch.multiply(A1_g, A1_g)) + scale2 * torch.sum(torch.multiply(A2_g, A2_g)) + scale3 * torch.sum(torch.multiply(A3_g, A3_g)) + scale4 * torch.sum(torch.multiply(A4_g, A4_g)) + scale5 * torch.sum(torch.multiply(A5_g, A5_g)) + scale6 * torch.sum(torch.multiply(A6_g, A6_g));
    
    # a = scale1 * torch.sum(torch.multiply(A1_g, res1)) + scale2 * torch.sum(torch.multiply(A2_g, res2)) + scale3 * torch.sum(torch.multiply(A3_g, res3)) + scale4 * torch.sum(torch.multiply(A4_g, res4)) + scale5 * torch.sum(torch.multiply(A5_g, res5)) + scale6 * torch.sum(torch.multiply(A6_g, res6));
    alpha = 1.0*a/b;
    # alpha = -1.0*a/b;
    
    return alpha 