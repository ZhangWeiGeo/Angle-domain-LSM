#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:20:05 2023

@author: zhangjiwei
"""

from Library import *
from make_data_3D import imshow_and_write_km as pw
from admm_func import apply_one_torch_operator
from admm_func import  Compute_L2_gradient_with_torch_operator_2D
from admm_func import  Compute_L2_step_length_with_torch_operator_2D
#from admm_func import *
   
class DEM_forward(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.2, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001, network_bool=3, Unrolling_way=1, output_bool=False):
        super().__init__()
        print("class DEM_forward(nn.Module)");
        print("in_channels=",in_channels);
        print("ou_channels=",ou_channels);
        print("filter_num=",filter_num);
        print("kernel_size=",kernel_size);
        print("padding=",padding);
        print("padding_mode=",padding_mode);
        print("is_deconv=",is_deconv);
        print("is_batchnorm=",is_batchnorm);
        print("spectral_norm=",spectral_norm);
        
        print("init_weight=",init_weight);
        print("bias_bool1=",bias_bool1);
        print("bias_bool2=",bias_bool2);
        print("Leaky=",Leaky);
        print("neu_in_num=",neu_in_num);
        print("neu_ou_num=",neu_ou_num);
        print("drop_layer=",drop_layer);
        print("drop_out=",drop_out);
        print("neu_ini=",neu_ini);
        
        # self.res_norm = []
        self.Unrolling_way = Unrolling_way;
        self.network_bool  = network_bool;
        self.drop_layer    = drop_layer;
        self.drop_out      = drop_out;
        
        if self.network_bool == 4:
            self.net = N_unet.UnetModel(in_channels, ou_channels, filter_num, 4, drop_out);
        
        if self.network_bool == 5:
            self.net = N_unet.DnCNN(in_channels);
            
            
        elif self.network_bool == 1:
            self.net = N.DEM_net_1(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);#U-net
        
        elif self.network_bool == 2:
            self.net = N.DEM_net_2(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);#Wu2019
        
        elif self.network_bool == 3:
            self.net = N.DEM_net_3(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);#Standard Conv with Batch normal
            
        elif self.network_bool == 30:
            self.net = N.DEM_net_30(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
            
        elif self.network_bool == 33:
            self.net = N.DEM_net_33(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
            
        elif self.network_bool == 31:
            self.net = N.DEM_net_31(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);    
        
        elif self.network_bool == 32:
            self.net = N.DEM_net_32(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
          
            
        elif self.network_bool == 40:
            self.net = N.DEM_net_40(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
        elif self.network_bool == 41:
            self.net = N.DEM_net_41(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
       
            
        elif self.network_bool == 50:
            self.net = N.DEM_net_50(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
        elif self.network_bool == 51:
            self.net = N.DEM_net_51(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
            
            
        elif self.network_bool == 300:
            self.net = N.DEM_net_300(in_channels, ou_channels, filter_num, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, is_deconv=is_deconv, is_batchnorm=is_batchnorm, spectral_norm=spectral_norm, init_weight=init_weight, bias_bool1=bias_bool1, bias_bool2=bias_bool2, Leaky=Leaky, neu_in_num=neu_in_num, neu_ou_num=neu_ou_num, drop_layer=drop_layer, drop_out=drop_out, neu_ini=neu_ini);
        
        if self.network_bool !=0:
            # Initialization of Parameters        
            # N.Initialization_net(self.net, init_weight, output_txt=False);
            self.net = self.net.to(device);
        
    def forward(self, y, x, For_ward, Adj_joint, check_point_bool=True, net_iter_input_nor=True):
        #y is obs, x is the model parameters, for a inverse problem
        # print("x.requires_grad", x.requires_grad)
        
        inv_grad       = torch.zeros_like(x, requires_grad=False);
        x_update       = torch.zeros_like(x, requires_grad=False);
        
        inv_sys        =  apply_one_torch_operator(For_ward, x);
        inv_sys_res    =  inv_sys - y ;
        inv_grad       =  Compute_L2_gradient_with_torch_operator_2D([Adj_joint,] , [inv_sys_res,], [1,]);
        alpha          =  Compute_L2_step_length_with_torch_operator_2D(inv_grad, [For_ward,],[inv_sys_res,],[1,]);
        
        x_update[:,:,:,:]   =  x[:,:,:,:] - 1.0 * alpha*inv_grad[:,:,:,:];

        if net_iter_input_nor:
            max_abs_val = torch.max( torch.abs(x) );
            if max_abs_val!=0:
                x    = x/max_abs_val; #this normalized is works in this function, it does not work out of this function for the revision of x; otherwise x[:] = x[:]/max_abs_val;

        if self.Unrolling_way==1:
            
            if check_point_bool:
                net_x = checkpoint( self.net, x, use_reentrant=False);
            else:
                net_x = self.net(x)
            output = x_update - alpha * net_x
            
            return output, x, x_update, alpha*inv_grad, alpha*net_x, torch.sum( torch.multiply(inv_sys_res,inv_sys_res) )
 
        elif self.Unrolling_way==2:
            
            if check_point_bool:
                output = checkpoint( self.net, x_update, use_reentrant=False);
            else:
                output=self.net(x_update)

            return output, x, x_update, alpha*inv_grad, alpha*inv_grad, torch.sum(torch.multiply(inv_sys_res,inv_sys_res))


# @torch.no_grad()
def forward_iteration_deep_un(f, x0, max_iter=100, tol=1e-3, forward=True, output_bool=False):
    # print("forward_iteration_deep_em max_iter is ",max_iter);
    # print("forward_iteration_deep_em tol is ",tol);
    # print("forward_iteration_deep_em output_bool is ",output_bool);
    # forward bool: forward f: we need to record the middle variables, backward f, we only compute the final result.
    # res1_arr is norm1(x2-x1), res2_arr:norm2( Ax2-Ax1), 

    res1_arr        = []
    
    f_arr           = [];
    x_arr           = [];
    x_update_arr    = [];
    inv_grad_arr    = [];
    net_x_arr       = [];
    res2_arr        = [];
    
    if forward:
        f0, x, x_update, inv_grad, net_x, res2 = f(x0);
        if output_bool:
            f_arr.append(f0)
            x_arr.append(x)
            x_update_arr.append(x_update)
            inv_grad_arr.append(inv_grad)
            net_x_arr.append(net_x)
        res2_arr.append( res2.item() )
        
    else:
        f0 = f(x0);
        
    
    for k in range(max_iter):
        
        f_exc = 1.0 * f0
        
        if forward:
            f0, x, x_update, inv_grad, net_x, res2 = f(f_exc);
            if output_bool:
                f_arr.append(f0)
                x_arr.append(x)
                x_update_arr.append(x_update)
                inv_grad_arr.append(inv_grad)
                net_x_arr.append(net_x)
            res2_arr.append( res2.item() )
            
        else:
            f0 = f(f_exc);
        
        res1_arr.append( (f0 - f_exc).norm().item() / (1e-5 + f0.norm().item()) )
        
        if (res1_arr[-1] < tol):
            break
    
    return f0, res1_arr, k, f_arr, x_arr, x_update_arr, inv_grad_arr, net_x_arr, res2_arr


class Deep_unrol_iteration(nn.Module):
    def __init__(self, f, solver, max_iter1=10, tol1=-1, max_iter2=10, tol2=-1, output_bool=False):
        super().__init__()
        self.f = f
        self.solver = solver
        # self.kwargs = kwargs
        self.max_iter1 = max_iter1
        self.tol1 = tol1
        self.max_iter2 = max_iter2
        self.tol2 = tol2
        self.output_bool = output_bool
        
    def forward(self, y, x, backward_mode=True):
        # f_true = lambda X : self.f(y, X);
        
        # compute forward pass and re-engage autograd tape
        # with torch.no_grad():
        x_last, self.forward_res1, self.total_iter1, self.f_arr, self.x_arr, self.x_update_arr, self.inv_grad_arr, self.net_x_arr, self.forward_res2 = self.solver(self.f, x, self.max_iter1, self.tol1, forward=True, output_bool=self.output_bool); #we have revise the initial value torch.zeros_like(x) as x
        
        return x_last


class DEQFixedPoint_return(nn.Module):
    def __init__(self, f, solver, max_iter1=100, tol1=1e-3, max_iter2=100, tol2=1e-3, output_bool=False):
        super().__init__()
        self.f = f
        self.solver = solver
        # self.kwargs = kwargs
        self.max_iter1 = max_iter1
        self.tol1 = tol1
        self.max_iter2 = max_iter2
        self.tol2 = tol2
        self.grad = [] #it does not free memory
        self.output_bool = output_bool
        
        
    def forward(self, y, x, backward_mode=True):
        # f_true = lambda X : self.f(y, X);
        
        # compute forward pass
        with torch.no_grad():
            x_last, self.forward_res1, self.total_iter1, self.f_arr, self.x_arr, self.x_update_arr, self.inv_grad_arr, self.net_x_arr, self.forward_res2 = self.solver(self.f, x, self.max_iter1, self.tol1, forward=True, output_bool=self.output_bool); #we have revise the initial value torch.zeros_like(x) as x
        #and re-engage autograd tape
        z, _, _, _, _, _   = self.f(x_last);
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0, _, _, _, _, _  = self.f(z0)
        
        # print("x_last.requires_grad is {}",f0.requires_grad)
        # print("z.requires_grad is {}",z.requires_grad)
        # print("z0.requires_grad is {}",z0.requires_grad)
        # print("f0.requires_grad is {}",f0.requires_grad)
        
        def backward_hook(grad):
            g, self.backward_res1, _, _, _, _, _, self.total_iter2 = self.solver(lambda beta : autograd.grad(f0, z0, beta, retain_graph=True)[0] + grad, grad, self.max_iter2, self.tol2, forward=False, output_bool=False) #We must setting forward=False, output_bool=False, see self.solver
            return g
        
        if backward_mode:
            # hook_f = lambda xx : T.hook_save(self.grad,xx)
            # ## 1. save the previous gradient
            # z.register_hook(hook_f)
            ## 2. modified z's gradient
            z.register_hook(backward_hook)
            ## 3. save the modified gradient
            # z.register_hook(hook_f)
        
        return z 
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # #########old
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


#@torch.no_grad()
def anderson(f, x0, m=5, lam=1e-2, max_iter=100, tol=1e-3, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    # X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,0], F[:,0] = 1.0*x0.reshape(bsz, -1), 1.0*f(x0).reshape(bsz, -1)
    X[:,1], F[:,1] = 1.0*F[:,0], 1.0*f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = 1.0*F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        # alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        alpha = torch.linalg.solve(H[:,:n+1,:n+1],y[:,:n+1])[:, 1:n+1, 0] 
        
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
        # print("k is {} res is {}".format(k,res[k-1]));
    return X[:,k%m].view_as(x0), res


#@torch.no_grad()
def forward_iteration(f, x0, max_iter=50, tol=1e-3):
    # print('x0', x0.device)
    f0 = f(x0);
    # print('f0', f0.device)
    res = []
    for k in range(max_iter):
        
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
        if (res[-1] < tol):
            break
        # print("k is {} res is {}".format(k,res[k]));
    return f0, res, k
  
  
class DEQFixedPoint(nn.Module):
    # def __init__(self, f, solver, **kwargs):
    def __init__(self, f, solver, max_iter1=100, tol1=1e-3, max_iter2=100, tol2=1e-3):
        super().__init__()
        self.f = f
        self.solver = solver
        # self.kwargs = kwargs
        self.max_iter1 = max_iter1
        self.tol1 = tol1
        self.max_iter2 = max_iter2
        self.tol2 = tol2
        self.grad = []
        
    def forward(self, y):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            x_last, self.forward_res1, self.total_iter1 = self.solver(self.f, torch.zeros_like(y), self.max_iter1, self.tol1);

        z = self.f(x_last);
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0)
        # print("x_last.requires_grad is {}",x_last.requires_grad)
        # print("z.requires_grad is {}",z.requires_grad)
        # print("z0.requires_grad is {}",z0.requires_grad)
        # print("f0.requires_grad is {}",f0.requires_grad)
        
        def backward_hook(grad):
            g, self.backward_res1, self.total_iter2 = self.solver(lambda beta : autograd.grad(f0, z0, beta, retain_graph=True)[0] + grad, grad, self.max_iter2, self.tol2)
            return g
        
        # hook_f = lambda xx : T.hook_save(self.grad,xx)
        # # 1. save the previous gradient
        # z.register_hook(hook_f)  
        # # 2. modified z's gradient
        z.register_hook(backward_hook)
        # # 3. save the modified gradient
        # z.register_hook(hook_f) 
        
        self.backward_res =0 ;
        self.total_iter2 = 0 ;
        
        return z
    
        
# =============================================================================
#     
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================

###mv function to admm_func.py

# =============================================================================
#             
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================
# =============================================================================
# 
# =============================================================================
def weight_and_bias_eps(w_arr, name, rows=4, figsize=(10, 5), color_arr=['seismic', 'gray'], eps_dpi=100):    
    
    num = len(w_arr)
    
    for j in range(0, len(color_arr)):
        
        for i in range(0, num):
            output = T.torch_to_np( w_arr[i].detach() );
            
            size1, size2, size3, size4 = output.shape
            id_x = int(size2 / 2 );
            
            plot_arr  = output[:, id_x, :, :];
            
            columns   = int ( np.floor(1.0*size1/rows) )
            
            fig = plt.figure(figsize=figsize)
            
            for ix in range(0, size1):
                m = plot_arr[ ix, :, :].reshape(size3, size4)
                if columns>1:
                    
                    fig.add_subplot(rows, columns, ix+1)
                    
                plt.imshow(m, cmap=color_arr[j], vmin=plot_arr.min(), vmax=plot_arr.max());
                plt.axis('off')
                plt.tight_layout()
                if (ix+1) % rows ==0:  
                    plt.colorbar(shrink=0.4)

            final_name = name + "-layer-" + str(i) + '-' + color_arr[j]+ ".eps"; 
            plt.savefig(final_name, dpi=eps_dpi)
            plt.close()


def output_hook_grad_eps(deq, iter_m, eps_path, eps_dpi=100):

    [_, _, nx, nz] = deq.grad[0].size()
    
    z_previous_grad = T.torch_to_np(deq.grad[0]).reshape((nx,nz));
    z_current_grad  = T.torch_to_np(deq.grad[1]).reshape((nx,nz));
        
    name="z-previous-grad-" + str(iter_m)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(z_previous_grad, eps_name, file_name);


    name="z-current-grad-" + str(iter_m)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(z_current_grad, eps_name, file_name);
    
    
    file='iteration={} z_previous_grad.max {}\n'.format(iter_m, np.max(z_previous_grad))
    print(file);P.write_txt(log_path+"log.txt", file,type='a+');
    
    file='iteration={} z_previous_grad.min {}\n'.format(iter_m, np.min(z_previous_grad))
    print(file);P.write_txt(log_path+"log.txt", file,type='a+');
        
    file='iteration={} z_current_grad.max {}\n'.format(iter_m, np.max(z_current_grad))
    print(file);P.write_txt(log_path+"log.txt", file,type='a+');
    
    file='iteration={} z_current_grad.min {}\n'.format(iter_m, np.min(z_current_grad))
    print(file);P.write_txt(log_path+"log.txt", file,type='a+');


def output_log1(iter_m, loss_value, loss1_value, loss2_value, loss3_value, tv_loss_value, l1_loss_value, loss_scale2, loss_scale3, loss1_coeff, loss2_coeff, loss3_coeff, lr_value, admm_s_max, f_cita_max, s_soft_value, y_soft_value, deq, DEM_or_DU, log_path, eps_dpi=100):
    #log
    file='iteration={} loss[{}]={}\n'.format(iter_m, iter_m, loss_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} loss1[{}]={}\n'.format(iter_m, iter_m, loss1_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} loss2[{}]={}\n'.format(iter_m, iter_m, loss2_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} loss3[{}]={}\n'.format(iter_m, iter_m, loss3_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} tv_loss[{}]={}\n'.format(iter_m, iter_m, tv_loss_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} l1_loss[{}]={}\n'.format(iter_m, iter_m, l1_loss_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} loss_scale2[{}]={}\n'.format(iter_m, iter_m, loss_scale2);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} loss_scale3[{}]={}\n'.format(iter_m, iter_m, loss_scale3);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} loss1_coeff[{}]={}\n'.format(iter_m, iter_m, loss1_coeff);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} loss2_coeff[{}]={}\n'.format(iter_m, iter_m, loss2_coeff);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} loss3_coeff[{}]={}\n'.format(iter_m, iter_m, loss3_coeff);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} lr[{}]={}\n'.format(iter_m, iter_m, lr_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} admm_s_max[{}]={}\n'.format(iter_m, iter_m, admm_s_max);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} s_soft_value[{}]={}\n'.format(iter_m, iter_m, s_soft_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} f_cita_max[{}]={}\n'.format(iter_m, iter_m, f_cita_max);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    file='iteration={} y_soft_value[{}]={}\n'.format(iter_m, iter_m, y_soft_value);
    print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
    if DEM_or_DU != 0:
        file='iteration={} total_iter1[{}]={}\n'.format(iter_m, iter_m, deq.total_iter1);
        print(file); P.write_txt(log_path+"log.txt", file, type='a+');
    
        file='iteration={} total_iter2[{}]={}\n'.format(iter_m, iter_m, deq.total_iter2);
        print(file); P.write_txt(log_path+"log.txt", file, type='a+');
        

def output_loss_eps(iter_m, loss1_arr, loss2_arr, loss3_arr, tv_loss_arr, l1_loss_arr, lr_arr, data_path, eps_path, log_path, eps_dpi=100):
    
    #loss and lr arrary
    name="loss1-" + str(iter_m)
    plt.semilogy(loss1_arr);
    plt.legend(['Training']);plt.xlabel("Iteration");plt.ylabel("L2 loss function");
    plt.savefig(eps_path+ name +'.eps', dpi=eps_dpi);plt.close();
    
    file_name = data_path + name + ".bin"
    P.fwrite_file_1d(file_name, np.asarray(loss1_arr), len(loss1_arr));
    
    
    name="loss2-" + str(iter_m)
    plt.semilogy(loss2_arr);
    plt.legend(['Training']);plt.xlabel("Iteration");plt.ylabel("ADMM for TV loss function");
    plt.savefig(eps_path+ name +'.eps', dpi=eps_dpi);plt.close();
    
    file_name = data_path + name + ".bin"
    P.fwrite_file_1d(file_name, np.asarray(loss2_arr), len(loss2_arr));
    
    
    name="loss3-" + str(iter_m)
    plt.semilogy(loss3_arr);
    plt.legend(['Training']);plt.xlabel("Iteration");plt.ylabel("ADMM for L1 Loss function");
    plt.savefig(eps_path+ name +'.eps', dpi=eps_dpi);plt.close();
    
    file_name = data_path + name + ".bin"
    P.fwrite_file_1d(file_name, np.asarray(loss3_arr), len(loss3_arr));
    
    
    name="tv-loss-" + str(iter_m)
    plt.semilogy(tv_loss_arr);
    plt.legend(['Training']);plt.xlabel("Iteration");plt.ylabel("TV Loss function");
    plt.savefig(eps_path+ name +'.eps', dpi=eps_dpi);plt.close();
    
    file_name = data_path + name + ".bin"
    P.fwrite_file_1d(file_name, np.asarray(tv_loss_arr), len(tv_loss_arr));
    
    
    name="l1-loss-" + str(iter_m)
    plt.semilogy(l1_loss_arr);
    plt.legend(['Training']);plt.xlabel("Iteration");plt.ylabel("L1 Loss function");
    plt.savefig(eps_path+ name +'.eps', dpi=eps_dpi);plt.close();
    
    file_name = data_path + name + ".bin"
    P.fwrite_file_1d(file_name, np.asarray(l1_loss_arr), len(l1_loss_arr));
    
    
    name="lr" + str(iter_m)
    plt.semilogy(lr_arr);
    plt.legend(['Training']);plt.xlabel("Iteration");plt.ylabel("Learning rater");
    plt.savefig(eps_path+ name +'.eps', dpi=eps_dpi);plt.close();
    
    file_name = data_path + name + ".bin"
    P.fwrite_file_1d(file_name, np.asarray(lr_arr), len(lr_arr));
   
    
def output_train_eps(iter_m, deq, data_path, eps_path, log_path, net_out_sys, net_out_sys_res, net_out, ou_y, ref_arr, inv_scale, forward_scale, eps_dpi=100):
        
    [_, _, nx, nz] = net_out_sys.size()
        
    mig_sys = inv_scale * forward_scale * T.torch_to_np(net_out_sys).reshape((nx,nz));
    mig_res = inv_scale * forward_scale * T.torch_to_np(net_out_sys_res).reshape((nx,nz));
    inv     = inv_scale * T.torch_to_np(net_out).reshape((nx,nz));
        
    if iter_m==0: 
        #ou_y
        ou_y    = inv_scale * forward_scale * T.torch_to_np(ou_y).reshape(nx,nz);
        name= "ou-y-" + str(nx) + "-"  + str(nz)
        file_name = data_path + name + ".bin"
        eps_name  = eps_path  + name
        pw(ou_y, eps_name, file_name, plot_min=ou_y.min(), plot_max=ou_y.max());
        
        
    # inv
    name="inv-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(inv, eps_name, file_name);
        
        
    # mig sys
    name="mig-sys-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m) 
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(mig_sys, eps_name, file_name, plot_min=ou_y.min(), plot_max=ou_y.max());
      
        
    # mig res
    name="mig-res-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)     
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(mig_res, eps_name, file_name, plot_min=ou_y.min(), plot_max=ou_y.max());
    
    
    #ref
    name="ref-" + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( ref_arr , eps_name, file_name, plot_min=ref_arr.min(), plot_max=ref_arr.max() );    
    
        
    # res 1
    name="inv-res1-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(inv - ref_arr, eps_name, file_name, plot_min=ref_arr.min(), plot_max=ref_arr.max() );
        
        
    # res 2
    name="inv-res2-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(ref_arr - inv , eps_name, file_name, plot_min=ref_arr.min(), plot_max=ref_arr.max());
        
        
    # res 3
    name="inv-res3-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(np.abs(inv - ref_arr) , eps_name, file_name, plot_min=ref_arr.min(), plot_max=ref_arr.max());
      
    

def output_forward_backward_eps(iter_m, deq, DEM_or_DU, data_path, eps_path, log_path, backward_model=False, eps_dpi=100):
    
    plt.semilogy(deq.forward_res1);
   
    if DEM_or_DU!=0 and backward_model:
        plt.semilogy(deq.backward_res1);
        plt.legend(['Forward', 'Backward']);
    else:
        plt.legend(['Forward']);
    
    plt.xlabel("Iteration");plt.ylabel("Relative Residual");
    plt.savefig(eps_path+'Norm2(f2-f1)-'+ str(iter_m)+'.eps', dpi=eps_dpi);plt.close();
    
    
    name = 'Norm2(f2-f1)-'+ str(iter_m) + "-forward"
    file_name = data_path + name + ".bin"
    P.fwrite_file_1d(file_name, np.asarray(deq.forward_res1), 1);
    
    if DEM_or_DU!=0 and backward_model:
        name = 'Norm2(f2-f1)-'+ str(iter_m) + "-backward"
        file_name = data_path + name + ".bin"
        P.fwrite_file_1d(file_name, np.asarray(deq.backward_res1), 1);
    
    
    #Image-domain matching item residual
    name="Norm2(Ax-y)-" + str(iter_m)
    plt.semilogy(deq.forward_res2);
    plt.legend([name]);plt.xlabel("Iteration");plt.ylabel("Objective function");
    plt.savefig(eps_path + name +'.eps', dpi=eps_dpi);plt.close();

    file_name = data_path + name + ".bin"
    P.fwrite_file_1d(file_name, np.asarray(deq.forward_res2), 1);



def output_deq_return_result_eps(For_ward, deq, iter_m, iter1, ou_y, data_path, eps_path, log_path, inv_scale, forward_scale, eps_dpi=100):

    # self.f_arr, self.x_arr, self.x_update_arr, self.inv_grad_arr, self.net_x_arr
    # [nx, nz]    =  deq.f_arr[iter1].size()  ##
    [_, _, nx, nz]    =  deq.f_arr[iter1].size()
    
    inv         = inv_scale * T.torch_to_np(deq.f_arr[iter1]).reshape((nx,nz));
    x           = T.torch_to_np(deq.x_arr[iter1]).reshape((nx,nz));
    x_update    = inv_scale * T.torch_to_np(deq.x_update_arr[iter1]).reshape((nx,nz));
    inv_part1   = inv_scale * T.torch_to_np(deq.inv_grad_arr[iter1]).reshape((nx,nz));
    inv_part2   = inv_scale * T.torch_to_np(deq.net_x_arr[iter1]).reshape((nx,nz));
    
    
    mig_sys     = inv_scale * forward_scale * apply_one_torch_operator(For_ward, deq.f_arr[iter1]);
    ou_y        = inv_scale * forward_scale * ou_y;
    mig_res     = mig_sys - ou_y;
    
    
    mig_sys = T.torch_to_np(mig_sys).reshape(nx,nz);
    mig_res = T.torch_to_np(mig_res).reshape(nx,nz);
    
    
    name="inv-" + str(iter_m) + "-" + str(iter1)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( inv , eps_name, file_name);
    
    
    name="inv-x-nor-" + str(iter_m) + "-" + str(iter1)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( x , eps_name, file_name);##normalized x
    
    
    name="inv-x-update-" + str(iter_m) + "-" + str(iter1)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( x_update , eps_name, file_name);
    
    
    name="inv-x-" + str(iter_m) + "-" + str(iter1)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( x_update + inv_part1 , eps_name, file_name);#no normalized x


    name="inv-part1-" + str(iter_m) + "-" + str(iter1)  + "-" + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( inv_part1, eps_name, file_name);
    
    
    name="inv-part2-" + str(iter_m) + "-" + str(iter1)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( inv_part2 , eps_name, file_name);
    
    
    
    name="mig-sys-" + str(iter_m) + "-" + str(iter1)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( mig_sys , eps_name, file_name, plot_min=ou_y.min(), plot_max=ou_y.max());
    
    
    name="mig-res-" + str(iter_m) + "-" + str(iter1)  + "-" + str(nx) + "-"  + str(nz)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( mig_res , eps_name, file_name, plot_min=ou_y.min(), plot_max=ou_y.max());


    
def output_predict_eps(deq, pred_out, ref_arr, data_path, eps_path, log_path, iter_m="predict", eps_dpi=100):
    
    [nx, nz] = pred_out.shape
    
    #predict
    name="pred-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( pred_out , eps_name, file_name);


    #pred-res1
    name="pred-res1-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( pred_out - ref_arr , eps_name, file_name);
    
    
    #pred-res2
    name="pred-res2-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( ref_arr - pred_out , eps_name, file_name);
    
    
    #pred res3
    name="pred-res3-" + str(nx) + "-"  + str(nz) + "-"  + str(iter_m)
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw( np.fabs(ref_arr - pred_out) , eps_name, file_name);
    
    
    
    
def output_uq_eps(pred_mean, pred_var, ref_arr, data_path, eps_path, log_path, color_arr, eps_dpi=100):
    
    [nx, nz] = pred_mean.shape
    
    #mean
    name="pred-" + "mean-" + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    for ix, color in enumerate( color_arr ):
        eps_name  = eps_path  + name + "-" + str(color);
        pw( pred_mean , eps_name, file_name, color=color);
    
    #var
    name="pred-" + "var-" + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    for ix, color in enumerate( color_arr ):
        eps_name  = eps_path  + name + "-" + str(color);
        pw( pred_var , eps_name, file_name, color=color);
    
    
    #ref
    name="ref-" + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    for ix, color in enumerate( color_arr ):
        eps_name  = eps_path  + name + "-" + str(color);
        pw( ref_arr , eps_name, file_name, color=color);
    
    
    #res 1
    name="pred-" + "mean-" + "res1-"  + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    for ix, color in enumerate( color_arr ):
        eps_name  = eps_path  + name + "-" + str(color);
        pw( pred_mean - ref_arr , eps_name, file_name, color=color, plot_min=ref_arr.min(), plot_max=ref_arr.max());

    
    #res 2
    name="pred-" + "mean-" + "res2-" + str(nx) + "-"  + str(nz)  
    file_name = data_path + name + ".bin"
    for ix, color in enumerate( color_arr ):
        eps_name  = eps_path  + name + "-" + str(color);
        pw( ref_arr - pred_mean , eps_name, file_name, color=color, plot_min=ref_arr.min(), plot_max=ref_arr.max());
    
    
    #res 3
    name="pred-" + "mean-" + "res3-" + str(nx) + "-"  + str(nz)  
    file_name = data_path + name + ".bin"
    for ix, color in enumerate( color_arr ):
        eps_name  = eps_path  + name + "-" + str(color);
        pw( np.fabs(ref_arr - pred_mean) , eps_name, file_name, color=color, plot_min=ref_arr.min(), plot_max=ref_arr.max());
        
    
    #res 4
    name="pred-" + "mean-" + "res4-" + str(nx) + "-"  + str(nz)  
    file_name = data_path + name + ".bin"
    for ix, color in enumerate( color_arr ):
        eps_name  = eps_path  + name + "-" + str(color);
        pw( np.fabs(ref_arr - pred_mean) , eps_name, file_name, color=color);

    
    
def compute_pred_mean_mig_res_eps(For_ward, pred_mean, ou_y, data_path, eps_path, log_path, color_arr, inv_scale, forward_scale, eps_dpi=100):
    
    [nx, nz]  = pred_mean.shape;
    pred_mean = pred_mean.reshape(1, 1, nx, nz); 
    
    pred_mean = T.np_to_torch(pred_mean).to(ou_y.device);
    
    ou_y      = forward_scale * inv_scale * ou_y;
    
    mig_sys   = forward_scale * apply_one_torch_operator(For_ward, pred_mean); ##pred_mean already with inv_scale
    mig_res   = mig_sys - ou_y; 
    
    mig_sys = T.torch_to_np(mig_sys).reshape(nx,nz);
    mig_res = T.torch_to_np(mig_res).reshape(nx,nz);
    
    ou_y    = T.torch_to_np(ou_y).reshape(nx,nz);
    
    #ou y
    name="ou-y-" + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(ou_y, eps_name, file_name, plot_min=ou_y.min(), plot_max=ou_y.max());
    
    
    #mig sys
    name="mig-sys-"  + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(mig_sys, eps_name, file_name, plot_min=ou_y.min(), plot_max=ou_y.max());
    
     
    #mig res
    name="mig-res-"  + str(nx) + "-"  + str(nz) 
    file_name = data_path + name + ".bin"
    eps_name  = eps_path  + name
    pw(mig_res, eps_name, file_name, plot_min=ou_y.min(), plot_max=ou_y.max());