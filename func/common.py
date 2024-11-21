#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 21:54:05 2023

@author: zhangw0c
"""

import numpy as np
import os
# import plot_func as P #from func import plot_func as P
import plot_func as P

def obtain_nnx_nny_nnz_from_nxnynz_wxwywz_3D(nx,ny,nz,wx,wy,wz):
    
    w1=wx;
    w2=wy;
    w3=wz;
    half_w1 = np.floor(w1/2);
    half_w2 = np.floor(w2/2);
    half_w3 = np.floor(w3/2);
    if (w1%2)==0 | (w2%2)==0  | (w3%2)==0:
        print("The width/height/y must be odd,w1=%d,w2=%d,w3=%d",w1,w2,w3);
        exit(0);
		
    s_n1 = np.floor((nx-half_w1)/w1);
    s_n2 = np.floor((ny-half_w2)/w2);
    s_n3 = np.floor((nz-half_w3)/w3);
    
    if((nx-half_w1)%w1==0):
        s_n1=s_n1;
    else:
        s_n1=s_n1+1;

    if((ny-half_w2)%w2==0):
        s_n2=s_n2;
    else:
        s_n2=s_n2+1;
			
    if((ny-half_w3)%w3==0):
        s_n3=s_n3;
    else:
        s_n3=s_n3+1;
    
    s_n1=np.int(s_n1+2);
    s_n2=np.int(s_n2+2);
    s_n3=np.int(s_n3+2);
    
    nnx=np.int(s_n1*	w1);
    nny=np.int(s_n2*	w2);
    nnz=np.int(s_n3*	w3);
    
    bbbl=w1;
    bbbf=w2;
    bbbu=w3;
    
    bbbr=nnx-nx-bbbl;
    bbbb=nny-ny-bbbf;
		
    bbbd=nnz-nz-bbbu;
    
    return nnx,nny,nnz,s_n1,s_n2,s_n3,bbbl,bbbf,bbbu,bbbr,bbbb,bbbd

# nnx,nny,nnz,s_n1,s_n2,s_n3,bbbl,bbbf,bbbu,bbbr,bbbb,bbbd=obtain_nnx_nny_nnz_from_nxnynz_wxwywz_3D(500,1,240,23,1,23);

            
def form_the_weight_coeffiecent_3D(nx,ny,nz,output_bool=0):

    coe_h=np.zeros((nx,ny,nz,8));
    for ix in range(0,nx):
        for iy in range(0,ny):
            for iz in range(0,nz):
                
                # id1=iz*nx*ny+iy*nx+ix;

                sx=1.0*ix/nx;
                sy=1.0*iy/ny;
                sz=1.0*iz/nz;

                coe_h[ix,iy,iz,0]=1.0*(1-sx)*(1-sy)*(1-sz);
                coe_h[ix,iy,iz,1]=1.0*  sx  *(1-sy)*(1-sz);
                coe_h[ix,iy,iz,2]=1.0*  sx  *(1-sy)*  sz  ;
                coe_h[ix,iy,iz,3]=1.0*(1-sx)*(1-sy)*  sz  ;
                coe_h[ix,iy,iz,4]=1.0*(1-sx)*  sy  *(1-sz);
                coe_h[ix,iy,iz,5]=1.0*  sx  *  sy  *(1-sz);
                coe_h[ix,iy,iz,6]=1.0*  sx  *  sy  *  sz  ;
                coe_h[ix,iy,iz,7]=1.0*(1-sx)*  sy  *  sz  ;
    
    if output_bool:
        
        P.mkdir("./psf-eps/");
        for imark in range(0,8):
            tmp=1.0*coe_h[:,0,:,imark];
            tmp=tmp.reshape(nx,nz)
            eps_name='./psf-eps/coe-' + str(imark) + '.eps'
            P.imshow(tmp.T,nx,nz,1,1,0,nx,10,0,nz,10,'x(grid)','y(grid)',tmp.min(),tmp.max(),1,'gray',(5,5),eps_name);
            
            P.mkdir("./psf-eps/coe/");
            file_name='./psf-eps/coe/' + str(imark) + '.bin'
            P.fwrite_file_2d(file_name,tmp.T,nx,nz);
            
    return coe_h

def expand_boundary_2D_zeros(psf_h,nx,nz,wx,wz,nnx,nnz,reverse=0):
    
    if reverse==0:
        wf_nxnynz1_h=1.0*np.zeros((nnx,nnz));
        wf_nxnynz1_h[wx:nx+wx,wz:nz+wz] = 1.0*psf_h[:,:];
    else:
        wf_nxnynz1_h = 1.0*psf_h[wx:nx+wx,wz:nz+wz];
    
    return wf_nxnynz1_h

def expand_boundary_2D_psf(psf_h,nx,nz,wx,wz,nnx,nnz):
    
    wf_nxnynz1_h=np.zeros((nnx,nnz));
    wf_nxnynz1_h[wx:nx+wx,wz:nz+wz] = 1.0*psf_h[:,:];

##up
    wf_nxnynz1_h[:,0:wz] = 1.0*wf_nxnynz1_h[:,wz:2*wz];
##down
    wf_nxnynz1_h[:,nnz-wz:nnz] = 1.0*wf_nxnynz1_h[:,nnz-2*wz:nnz-wz];
##left    
    wf_nxnynz1_h[0:wx,:] = 1.0*wf_nxnynz1_h[wx:2*wx,:];
##right    
    wf_nxnynz1_h[nnx-wx:nnx,:] = 1.0*wf_nxnynz1_h[nnx-2*wx:nnx-wx,:]
    
    return wf_nxnynz1_h

def reshape_2D_psf(mig,n1,n2,wx,wz,output_bool=0):
    
    output=np.zeros((wx,wz,n1,n2))
    
    for ix in range(0,n1):
        for iz in range(0,n2):

            begx=(ix+0)*wx
            endx=(ix+1)*wx
            begz=(iz+0)*wz
            endz=(iz+1)*wz
            output[:,:,ix,iz] = 1.0*mig[begx:endx,begz:endz];
            
            if output_bool:
                tmp=1.0*mig[begx:endx,begz:endz];
                P.mkdir("./psf-eps/");
                eps_name='./psf-eps/' + str(ix) + '-' + str(iz) + '.eps'
                P.imshow(tmp.T,wx,wz,1,1,0,wx,10,0,wz,10,'x(grid)','y(grid)',tmp.min(),tmp.max(),1,'gray',(5,5),eps_name);
                
                P.mkdir("./psf-eps/psf/");
                file_name='./psf-eps/psf/' + str(ix) + '-' + str(iz) + '.bin'
                P.fwrite_file_2d(file_name,tmp.T,wx,wz);

                nx,nz = mig.shape
                file_name='./psf-eps/psf/' + "psf-" + str(nx) + "-" + str(nz) + '.bin'
                P.fwrite_file_2d(file_name,mig.T,nx,nz);
    return output 


def cal_angle_ref_func(vel_arr, den_arr, angle_start=0, angle_num=90, dangle=1.0):
    
    (nx,nz) = vel_arr.shape
    angle_ref_arr = np.zeros((angle_num, nx, nz));###no dtype for float 64, it is better for use to use the float 64 for this function. Because there is an error/nan for vel_arr.dtype
   
    for ig in range(0, angle_num):
        for iz in range(0, nz-1):
            
            gama0 	  =   (angle_start + ig*dangle) /180.0*np.pi;
            c0        =   vel_arr[:,iz];
            c1        =   vel_arr[:,iz+1];
            rou0      =   den_arr[:,iz];
            rou1      =   den_arr[:,iz+1];
            
            radian 	  =  c1/c0*np.sin( gama0 );
            
            radian    = np.clip(radian, -1.0, 1.0) ;
            
            gama1 	=  ( 1.0*np.arcsin( radian )  );
            
            tmp0 	= rou1*c1*np.cos(1.0*gama0) - rou0*c0*np.cos(1.0*gama1) ; 

            tmp1 	= rou1*c1*np.cos(1.0*gama0) + rou0*c0*np.cos(1.0*gama1) ;

            final   = (1.0*tmp0 / tmp1).reshape(nx);

            mask = np.isnan(final)
            final[mask] = 0;            #print("mask is nan",mask);

            angle_ref_arr[ig,:,iz] =  final[:];
    
    return angle_ref_arr.astype(np.float32)


def expand_psfs_6D(psfs):
    (nx,ny,nz,wx,wy,wz) = psfs.shape
    output = np.zeros((nx+2, ny+2, nz+2, wx, wy, wz), dtype=psfs.dtype);
    
    output[1:nx+1, 1:ny+1, 1:nz+1, :, :, :] = 1.0*psfs[:, :, :, :, :, :];
    #x direction
    output[0:1, 1:ny+1, 1:nz+1, :, :, :]        = 1.0*psfs[0, :, :, :, :, :];
    output[nx+1:nx+2, 1:ny+1, 1:nz+1, :, :, :]  = 1.0*psfs[nx-1, :, :,  :, :, :];
    
    #y direction
    output[:,  0, :, :, :, :]    = 1.0*output[:,  1, :,  :, :, :]
    output[:, -1, :, :, :, :]    = 1.0*output[:, -2, :, :, :, :].reshape(nx+2, nz+2, wx, wy, wz);
    
    #z direction
    output[:, :,  0, :, :, :]      = 1.0*output[:, :,  1, :, :, :]
    output[:, :, -1, :, :, :]      = 1.0*output[:, :, -2, :, :, :].reshape(nx+2, ny+2, wx, wy, wz);
    
    return output

def find_psf_value_id(psfx, a):
    return_px = 0
    return_ipx   = 0
    for ipx, px in enumerate(psfx):
        if ipx<len(psfx)-1:
            if a>=px and a< psfx[ipx+1]:
                return_ipx   = ipx
                return_px = px
    
    if a>=psfx[-1]:
        return_ipx   = len(psfx)-1
        return_px    = psfx[-1]
    
    return return_ipx, return_px

def get_psf_id(total_psfx, part_psfx, multiple_psf_wx):
    
    psfx  = []
    t_idx = []
    
    for ipx, px in enumerate(part_psfx):
        idx, value = find_psf_value_id(total_psfx, px);
        if px<=total_psfx[-1]:
            psfx.append(value - px + ipx*multiple_psf_wx);
            t_idx.append(idx);
        else:
            psfx.append(  psfx[-1] + multiple_psf_wx);
            t_idx.append(t_idx[-1]);

    return psfx, t_idx

def convert_5D_psfs_to_3D_array(psfs):
    
    shape_list = list(psfs.shape);
    print("shape_list is ", list(shape_list));
    
    angle_num = shape_list[0]
    nx_number = shape_list[1]
    nz_number = shape_list[2]
    wx        = shape_list[3]
    wz        = shape_list[4]
    
    nx = wx *  nx_number
    nz = wz *  nz_number
    
    output = np.zeros((angle_num, nx, nz),dtype=psfs.dtype);
    
    for ix in range(0, nx_number):
        for iz in range(0, nz_number):
            output[:, ix*wx:(ix+1)*wx, iz*wz:(iz+1)*wz] = psfs[:,ix,iz,:,:].reshape(angle_num, wx, wz);
            
    return output

def convert_4D_psfs_to_2D_array(psfs):
    
    shape_list = list(psfs.shape);
    print("shape_list is ", list(shape_list));
    
    nx_number = shape_list[0]
    nz_number = shape_list[1]
    wx        = shape_list[2]
    wz        = shape_list[3]
    
    nx = wx *  nx_number
    nz = wz *  nz_number
    
    output = np.zeros((nx, nz),dtype=psfs.dtype);
    
    for ix in range(0, nx_number):
        for iz in range(0, nz_number):
            output[ix*wx:(ix+1)*wx, iz*wz:(iz+1)*wz] = psfs[ix,iz,:,:].reshape(wx, wz);
            
    return output

def compute_illumination_for_psfs_2D(mig_ones, psfx, psfz, illumination_wx, illumination_wz):
    (nx, nz) = mig_ones.shape
    output = np.zeros((len(psfx),len(psfz)),dtype=mig_ones.dtype);

    for ipx, px in enumerate(psfx):
        for ipz, pz in enumerate(psfz):
            
            begx = px   - illumination_wx
            if begx < 0:
                begx=0; 
                endx = begx + 2*illumination_wx
            else:
                endx = begx + illumination_wx
            if endx >nx-1:
                endx=nx-1;
                begx=endx-2*illumination_wx


            begz = pz   - illumination_wz
            if begz < 0:
                begz=0;
                endz = begz + 2*illumination_wz
            else:
                endz = begz + illumination_wz   
            if endz >nz-1:
                endz=nz-1;
                begz=endz-2*illumination_wz
            
            arr = mig_ones[begx:endx, begz:endz];
            
            mean_v = np.mean(arr);
            
            output[ipx,ipz] = 1.0*mean_v;
        
    return output