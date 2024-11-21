#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 10:40:41 2023

@author: zhangw0c
"""



import torch

def np_to_torch(np_arr):
    torch_arr=torch.from_numpy(np_arr);
    output = torch_arr.type(torch.FloatTensor);
    return output


def torch_to_np(torch_arr):
    input_tensor = torch_arr.to('cpu');
    output = input_tensor.detach().numpy();
    return output

def torch_detach_to_np(torch_arr):
    input_tensor = torch_arr.to('cpu');
    output = input_tensor.detach().numpy();
    return output

# ===================
# ===================
# how to use: hook_save in main function
# z_grad=[] ;                                       
# list for save gradient
# hook_func = lambda a: hook_save(z_grad,a)         
# define a function, z_grad is the defalut input, a is input of hook_func
# ref_arr_torch.register_hook(hook_func)            
# variable.register_hook means to the input is it's gradient
# mig_forward_pytorch.backward(mig_forward_pytorch);
#variable.backward
def hook_save(z_grad,grad):
    print(grad)
    z_grad.append(grad)
    # return z_grad
    
    