#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 09:27:06 2023

@author: zhangw0c
"""

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
    
    P.imshow1(input_arr.T,  x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=(6, 3.5), output_name=eps_name + fig_type, eps_dpi=eps_dpi, title=title, xtop=True, pltshow=False);
    
    P.fwrite_file(data_name, input_arr);


# input data from path       
psf_arr = np.zeros( (wangle, angle_num, nx, nz), dtype=np.float32 );
mig_arr = np.zeros( (angle_num, nx, nz), dtype=np.float32 );
ref_arr = np.zeros( (nx, nz), dtype=np.float32 );
vel_arr = np.zeros( (nx, nz), dtype=np.float32 );
den_arr = np.zeros( (nx, nz), dtype=np.float32 );


if multiple_psf_bool==0:
    for i in range(0,  min(len(psf_path), wangle) ):
        P.fread_file(psf_path[i], mig_arr, [angle_num, nx, nz]);
        psf_arr[i,:,:,:] = 1.0*mig_arr
else:
    total_psf_arr = np.zeros( (angle_num, psf_nx_number, psf_nz_number, wangle, wx, wz), dtype=np.float32 );
    tmp_arr       = np.zeros( (wangle, angle_num, psf_nx_number, wx, wz), dtype=np.float32 ); #0:wangle, 1:angle_num, 2:nx_number,3:wx,4:wz; 1, 3, 2, 0, 4, 5  psf_nz_number psf_nz_number
    
    t_start = time.perf_counter();
    for iz in range(0, psf_nz_number):
        
        z_pos = psf_start_z + iz*psf_interval_z
        file_name = psf_path + "-iz-" + str(z_pos)
        P.fread_file(file_name, tmp_arr, shape_list=(wangle, angle_num, psf_nx_number, wx, wz), way=0);
        
        total_psf_arr[:,:,iz,:,:,:] = 1.0 * np.transpose(tmp_arr, axes=(1, 2, 0, 3, 4));
    
    t_end = time.perf_counter();
    print(" psf has been read using one thread way=0 s)" + str((t_end - t_start)))


P.fread_file(mig_path, mig_arr, [angle_num, nx, nz]);
P.fread_file(ref_path, ref_arr, [nx, nz]);
P.fread_file(vel_path, vel_arr, [nx, nz]);
P.fread_file(den_path, den_arr, [nx, nz]);


name='ini-ref-' + str(nx) + "-"  + str(nz)
eps_name   =  eps_path  + "ini/" + name
data_name  = data_path + "ini/" + name + ".bin"
imshow_and_write_km(ref_arr, eps_name, data_name, full_model=True, velocity=False);


name='ini-vel-' + str(nx) + "-"  + str(nz)
eps_name   =  eps_path  + "ini/" + name
data_name  = data_path + "ini/" + name + ".bin"
for ix, color in enumerate( color_arr ):
    eps_name   =  eps_path  + "ini/" + name + "-" + str(color)
    imshow_and_write_km(vel_arr, eps_name, data_name, full_model=True, velocity=True, color=color);

name='ini-den-' + str(nx) + "-"  + str(nz)
eps_name   =  eps_path  + "ini/" + name
data_name  = data_path + "ini/" + name + ".bin"
for ix, color in enumerate( color_arr ):
    eps_name   =  eps_path  + "ini/" + name + "-" + str(color)
    imshow_and_write_km(den_arr, eps_name, data_name, full_model=True, density=True, color=color);
    
    
#calculate the angle dependent reflectivity
if len(angle_ref_name)==0:
    angle_ref_arr = C.cal_angle_ref_func(vel_arr, den_arr, angle_start=angle_start, angle_num=angle_num, dangle=dangle);
else:
    angle_ref_arr = np.zeros((angle_num, nx, nz), dtype=np.float32);
    P.fread_file(angle_ref_path, angle_ref_arr, [angle_num, nx, nz]);
    

name='ini-angle-ref-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, angle_ref_arr);


# =============================================================================
# ##cut ##cut##cut##cut##cut##cut##cut##cut##cut##cut##cut
# =============================================================================
if multiple_psf_bool!=0:
    total_psf_arr  = total_psf_arr[id_angle_beg:id_angle_end, :,  :, :, :, :];
    
psf_arr         = psf_arr[:, id_angle_beg:id_angle_end, x_beg:x_end, z_beg:z_end];
mig_arr         = mig_arr[id_angle_beg:id_angle_end,    x_beg:x_end, z_beg:z_end];
angle_ref_arr   = angle_ref_arr[id_angle_beg:id_angle_end, x_beg:x_end, z_beg:z_end];
ref_arr         = ref_arr[x_beg:x_end, z_beg:z_end];
vel_arr         = vel_arr[x_beg:x_end, z_beg:z_end];
den_arr         = den_arr[x_beg:x_end, z_beg:z_end];
[angle_num, nx, nz]    = mig_arr.shape
##############################################################################

name='ref-' + str(nx) + "-"  + str(nz)
eps_name   =  eps_path  + "ini/" + name
data_name  =  data_path + "ini/" + name + ".bin"
imshow_and_write_km(ref_arr, eps_name, data_name);


name='vel-' + str(nx) + "-"  + str(nz)
data_name  = data_path + "ini/" + name + ".bin"
for ix, color in enumerate( color_arr ):
    eps_name   =  eps_path  + "ini/" + name + "-" + str(color)
    imshow_and_write_km(vel_arr, eps_name, data_name, velocity=True, color=color);

name='den-' + str(nx) + "-"  + str(nz)
data_name  = data_path + "ini/" + name + ".bin"
for ix, color in enumerate( color_arr ):
    eps_name   =  eps_path  + "ini/" + name + "-" + str(color)
    imshow_and_write_km(den_arr, eps_name, data_name, density=True, color=color);


file1 = "true angle_num is {}, true nx is {}, true nz is{}\n".format(angle_num, nx, nz);
print(file1);  P.write_txt(log_path + "log.txt", file1+file1+file1, type='a+');




# define the difference operator
op_dx  = FirstDerivative(dims=(angle_num, nx, nz), axis=0, dtype=np.float32)  #, order=3
op_dx2 = SecondDerivative(dims=(angle_num, nx, nz), axis=0, dtype=np.float32)  #, order=3
op_dy  = FirstDerivative(dims=(angle_num, nx, nz), axis=1, dtype=np.float32)  #, order=3
op_dz  = FirstDerivative(dims=(angle_num, nx, nz), axis=2, dtype=np.float32)  #, order=3

op_dx_torch   = TorchOperator(op_dx,   device=device)
op_dx_torch_H = TorchOperator(op_dx.H, device=device)

op_dx2_torch   = TorchOperator(op_dx2,   device=device)
op_dx2_torch_H = TorchOperator(op_dx2.H, device=device)

op_dy_torch   = TorchOperator(op_dy,   device=device)
op_dy_torch_H = TorchOperator(op_dy.H, device=device)

op_dz_torch   = TorchOperator(op_dz,   device=device)
op_dz_torch_H = TorchOperator(op_dz.H, device=device)

# tmp_z = op_dz(mig_arr);
# tmp_y = op_dy(mig_arr);
# tmp_x = op_dx(mig_arr);



# set zeros for psfs
if psf_boundary_set_zeros:
    psf_arr[:,:, :, (nz//wz)*wz : nz] = 0.0
    psf_arr[:,:, (nx//wx)*wx : nx, :] = 0.0
    
    mig_arr[:, :, (nz//wz)*wz : nz] = 0.0
    mig_arr[:, (nx//wx)*wx : nx, :] = 0.0
    
    angle_ref_arr[:, :, (nz//wz)*wz : nz] = 0.0
    angle_ref_arr[:, (nx//wx)*wx : nx, :] = 0.0
    
    ref_arr[:, (nz//wz)*wz : nz] = 0.0
    ref_arr[(nx//wx)*wx : nx, :] = 0.0

name='mig-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  = data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, mig_arr);

name='angle-ref-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  = data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, angle_ref_arr);

name='psf-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  = data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, psf_arr[int(wangle//2),:,:,:]);

# reshape psf_arr to psfs, 
# if psf_arr is obtained at the position of discrete point of (wx,wz) from numerical modeling and migration operator or analytical modeling and migration
if multiple_psf_bool==0:
    # Psfx, Psfz = np.meshgrid(psfx, psfz, indexing='ij')
    psfa = np.arange(0, angle_num, 1)
    psfx = np.arange(wx//2, nx, wx)
    psfz = np.arange(wz//2, nz, wz)
    
    psfs=np.zeros( (len(psfa), len(psfx), len(psfz), wangle, wx, wz), dtype=np.float32)
    
    for iax, ax in enumerate(psfa):    
        for ipx, px in enumerate(psfx):
            for ipz, pz in enumerate(psfz):
                begx=int( px - wx//2     );
                endx=int( px + wx//2 + 1 );
                begz=int( pz - wz//2     );
                endz=int( pz + wz//2 + 1 );
                if endx<=nx and endz<=nz:
                    psfs[iax, ipx, ipz, :, :, :] = psf_arr[:, iax, begx:endx, begz:endz]
                else:
                    psfs[iax, ipx, ipz, :, :, :] = 0.0;
                
                # name = "psf-" + "ax-" + str(ax) +  "-" +  "px-"  + str(px) +  "-" +  "pz-"  + str(pz)
                # data_name  = data_path + "ini/" + name + ".bin"   
                # P.fwrite_file( data_name, psfs[iax, ipx, ipz, :, :, :]);
        
    if psf_boundary_copy:
        psfa = np.arange(0, angle_num+2, 1)
        psfx = np.arange(-wx//2, nx+wx, wx)
        psfz = np.arange(-wz//2, nz+wz, wz)
        psfs = C.expand_psfs_6D(psfs);

else:
    print(" total_psf_arr.shape is ",  total_psf_arr.shape );
    diff_len = int( (wangle - multiple_psf_wangle)/2);
    wangbeg = diff_len; wangend = multiple_psf_wangle + diff_len;
    print("wangbeg is {} wangend is {}".format(wangbeg, wangend) );
    
    psfa = np.arange(0, angle_num, 1);
    # total psf idx idz 
    total_psfx = np.arange(psf_start_x, psf_start_x + psf_nx_number*psf_interval_x,  psf_interval_x)
    total_psfz = np.arange(psf_start_z, psf_start_z + psf_nz_number*psf_interval_z,  psf_interval_z)
    
    P.write_np_txt(log_path + "x total_psfx.txt",  total_psfx)
    P.write_np_txt(log_path + "z total_psfz.txt",  total_psfz)


#get psfs at show_wx show_wz for show
if multiple_psf_bool!=0:
    
    # part psf idx idz
    part_psfx = np.arange(x_beg + show_wx//2, x_beg + nx, show_wx)
    part_psfz = np.arange(z_beg + show_wz//2, z_beg + nz, show_wz)
    
    psfs=np.zeros( (len(psfa), len(part_psfx), len(part_psfz), multiple_psf_wangle, wx, wz), dtype=np.float32)
    
    psfx, t_idx = C.get_psf_id(total_psfx, part_psfx, show_wx);

    psfz, t_idz = C.get_psf_id(total_psfz, part_psfz, show_wz);

    #obtain psfs at t_idx and t_idz
    for ipx, px in enumerate(part_psfx):
        for ipz, pz in enumerate(part_psfz):
            idx = t_idx[ipx]
            idz = t_idz[ipz]
            psfs[:, ipx, ipz, :, :, :] = 1.0* total_psf_arr[:, idx, idz, wangbeg:wangend, :, :];
    
    for i in range(0, multiple_psf_wangle):
        output = C.convert_5D_psfs_to_3D_array(psfs[:, :, :, i, :, :]);
        file = P.return_bin_name(output)
        name=str( int(i-multiple_psf_wangle//2) ) + '-psf-' + file
        data_name  = data_path + "ini/" + name + ".bin"
        P.fwrite_file(data_name, output);
    
    output=None;
    psfs=None;
    
#get psfs at multiple_psf_wx multiple_psf_wz     
if multiple_psf_bool!=0:
    
    # part psf idx idz
    part_psfx = np.arange(x_beg, x_end, multiple_psf_wx)
    part_psfz = np.arange(z_beg, z_end, multiple_psf_wz)
    
    psfs=np.zeros( (len(psfa), len(part_psfx), len(part_psfz), multiple_psf_wangle, wx, wz), dtype=np.float32)
    
    psfx, t_idx = C.get_psf_id(total_psfx, part_psfx, multiple_psf_wx);
   
    psfz, t_idz = C.get_psf_id(total_psfz, part_psfz, multiple_psf_wz);

    #obtain psfs at t_idx and t_idz
    for ipx, px in enumerate(part_psfx):
        for ipz, pz in enumerate(part_psfz):
            idx = t_idx[ipx]
            idz = t_idz[ipz]
            psfs[:, ipx, ipz, :, :, :] = 1.0* total_psf_arr[:, idx, idz, wangbeg:wangend, :, :];
            
            # name = "psf-" +  "px-"  + str(int(px)) +  "-" +  "pz-"  + str(int(pz))
            # data_name  = data_path + "ini/" + name + ".bin"   
            # P.fwrite_file( data_name, psfs[:, ipx, ipz, :, :, :]);
    
    
    P.write_np_txt(log_path + "x part_psfx.txt",  part_psfx)
    P.write_np_txt(log_path + "x psfx.txt",  psfx)
    P.write_np_txt(log_path + "x t_idx.txt",  t_idx)
    
    P.write_np_txt(log_path + "z part_psfz.txt",  part_psfz)
    P.write_np_txt(log_path + "z psfz.txt",  psfz)
    P.write_np_txt(log_path + "z t_idz.txt",  t_idz)

    # psfx = np.array(psfx) + error_x_length; psfz = np.array(psfz) + error_z_length;


P.write_np_txt(log_path + "final psfx.txt",  psfx);
P.write_np_txt(log_path + "final psfz.txt",  psfz);
psfs_shape_list = list(psfs.shape)
file0="size of psfs is {} G\n".format( psfs_shape_list[0]*psfs_shape_list[1]*psfs_shape_list[2]*psfs_shape_list[3]*psfs_shape_list[4]*psfs_shape_list[5]*4/1024.0/1024.0/1024.0);
file1="psfs.shape() is {}, psfs.max() is {}, psfs.min() is {}\n".format( psfs.shape, psfs.max(), psfs.min());
print(file0 + file1);P.write_txt(log_path + "log.txt", file0 + file1, type='a+');


forward_scale=1.0;
if psf_value_normalized:
    forward_scale=np.max(np.abs(psfs))
    psfs = psfs/forward_scale;
    file1="psfs.shape() is {}, psfs.max() is {}, psfs.min() is {}\n".format( psfs.shape, psfs.max(), psfs.min());
    print(file1);P.write_txt(log_path + "log.txt", file1, type='a+');
    

# 2D test only use the dignal angle to simulate the systhetic data
mig_sys = np.zeros_like(angle_ref_arr);
for i in range(0, angle_num):
    diag_psfs = psfs[i, :, :, int(multiple_psf_wangle/2), :, :].reshape(psfs_shape_list[1],psfs_shape_list[2],psfs_shape_list[4],psfs_shape_list[5]);
    # #numpy
    # Cop_cuda_2D = NonStationaryConvolve2D(dims=(nx, nz), hs=(diag_psfs.astype(np.float32)), ihx=psfx, ihz=psfz, engine="numba", dtype=np.float32 )
    # mig_sys[i,:,:]  =(Cop_cuda_2D(  (angle_ref_arr[i,:,:]) ));
    
    #cuda
    Cop_cuda_2D = NonStationaryConvolve2D(dims=(nx, nz), hs=cp.asarray(diag_psfs.astype(np.float32)), ihx=psfx, ihz=psfz, engine="cuda", dtype=np.float32 )
    mig_sys[i,:,:]  = forward_scale*cp.asnumpy(Cop_cuda_2D(  cp.asarray(angle_ref_arr[i,:,:]) ));

name='mig-sys-diag-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, mig_sys);

mig_res    = mig_sys - mig_arr;
name='mig-res-diag-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, mig_res);



#3D NonStationaryConvolve3D
from ID_cuda import NonStationaryConvolve3D
from torchoperator import TorchOperator

Cop_cuda = NonStationaryConvolve3D(dims=(angle_num, nx, nz), hs=cp.asarray(psfs).astype(np.float32), ihx=psfa, ihy=psfx, ihz=psfz, engine="cuda", dim_block= (8, 8, 8), dtype=np.float32 )



IDLSM_for_op    = TorchOperator(Cop_cuda._forward, Cop_cuda._adjoint, device=cp.zeros(10).device, devicetorch=device)
IDLSM_adj_op    = TorchOperator(Cop_cuda._adjoint, Cop_cuda._forward, device=cp.zeros(10).device, devicetorch=device)

t_start = time.perf_counter();
for i in range(0, 1):
    print("i is ", i);
    mig_sys 	    = forward_scale*Cop_cuda._forward(  cp.asarray(angle_ref_arr) );
    mig_sys_adj 	= forward_scale*Cop_cuda._adjoint(  cp.asarray(mig_sys) );
t_end = time.perf_counter();
file1="1 times forward and adjoint operator times is  {} \n".format( t_end-t_start );
P.write_txt(log_path + "log.txt",  file1, type='a+');print(file1);

mig_sys     	= cp.asnumpy(mig_sys);
mig_sys_adj	    = cp.asnumpy(mig_sys_adj);
mig_ones        = forward_scale*cp.asnumpy( Cop_cuda._forward(  cp.asarray(np.ones_like(angle_ref_arr)) ) );

name='mig-sys-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, mig_sys);

name='mig-sys-adj-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, mig_sys_adj);

name='mig-sys-ones-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, mig_ones);

# the residual between H_app*m and LTLm, just used to illustrate the accuracy of Hessian matrix through the point spread functions. The true valve for relfectivity inversion is not reflectivity!!! it's reference value is angle-dependent reflectivity *cos(r)^2
mig_res    = mig_sys - mig_arr;
name='mig-res-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, mig_res);


mig_res     = mig_sys/np.max(np.abs(mig_sys)) - mig_arr/np.max(np.abs(mig_arr)) ;
name='mig-res-nor-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, mig_res);


# dot test dot test
d1      = cp.random.rand( angle_num, nx,nz ).astype(np.float32) ;
m1      = cp.random.rand( angle_num, nx,nz ).astype(np.float32) ;
dot1    = cp.sum( Cop_cuda._forward( m1 )   * d1 ) ;
dot2    = cp.sum( Cop_cuda._adjoint( d1 )   * m1 ) ;

file1="dot1 is {}, dot2 is {} \n".format( dot1, dot2 );
print(file1);P.write_txt(log_path + "log.txt",  file1, type='a+');
# np.allclose(mmigforw, mmigforw_cuda), np.allclose(mmigadj, mmigadj_cuda)



# re-migration image of conventional migration image for supervised deep-learning
mig_arr_mig      = forward_scale*Cop_cuda._forward( cp.asarray(mig_arr).astype(np.float32) );
mig_arr_mig      = cp.asnumpy(mig_arr_mig) ;

name='mig-remig-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
P.fwrite_file(data_name, mig_arr_mig);



#input data
# X = torch.randn(1,1,nx,nz)
mig_arr           = mig_arr.reshape((1,angle_num,nx,nz));
# mig_arr_mig     = mig_arr_mig.reshape((1,angle_num,nx,nz));
mig_sys 	      = mig_sys.reshape((1,angle_num,nx,nz));

mig_torch 		  = T.np_to_torch(mig_arr).to(device);
# mig_remig_torch = T.np_to_torch(mig_arr_mig).to(device);
mig_crime_torch = T.np_to_torch(mig_sys).to(device);


# supervised
# in_x = 1.0*mig_remig_torch;
# ou_y = 1.0*mig_torch;
# unsupervised
in_x = 1.0*mig_torch;
ou_y = 1.0*mig_torch;
#in_x = 1.0 * mig_crime_torch;
#ou_y = 1.0 * mig_crime_torch;#inverse crime test


# normalzied aafter L2 inversion
if normalized_x:
    in_x = in_x / torch.max(torch.abs(in_x));
if normalized_y:
    inv_scale = 1.0 * torch.max(torch.abs(ou_y)).item()
    ou_y = ou_y / torch.max(torch.abs(ou_y));
    inv_scale = inv_scale/forward_scale
else:
    inv_scale = 1.0/forward_scale;

file1="when psf_value_normalized=True,normalized_y=True, we need to rescale the inverted result with inv_scale and rescale the sys-mig with inv_scale and forward_scale (inv_scale for the inverted result and forward_scale for forward operator)";
print(file1);P.write_txt(log_path + "log.txt",  file1, type='a+');

file1="psf_value_normalized={},normalized_y={}".format(psf_value_normalized, normalized_y);
print(file1);P.write_txt(log_path + "log.txt",  file1, type='a+');

file1="forward_scale={},inv_scale={}".format(forward_scale, inv_scale);
print(file1);P.write_txt(log_path + "log.txt",  file1, type='a+');


name='in-x-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, T.torch_to_np(in_x));

name='ou-y-' + str(angle_num) + "-"  + str(nx) + "-"  + str(nz)
data_name  =  data_path + "ini/" + name + ".bin"
P.fwrite_file(data_name, T.torch_to_np(ou_y));
