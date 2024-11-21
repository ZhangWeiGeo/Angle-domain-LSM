#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:20:12 2020

@author: zhangjiwei
"""

import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

##nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
 # Kernel size: 3*3, Stride: 1, Padding: 1
############ 3*3 conv  +  Prelu   +   BN
# self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1,1),nn.BatchNorm2d(out_size),nn.ReLU(inplace=True),)


####one time conv2d  BN prelu
class Conv2d_BN_Prelu(nn.Module):
    def __init__(self, in_size, out_size,filter_size=[3,3],stride_size=[1,1], is_batchnorm=False):
        super(Conv2d_BN_Prelu, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.PReLU(num_parameters=1,init=0.25))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.PReLU(num_parameters=1,init=0.25))
                                       
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

####two time conv2d  BN prelu
class Conv2d_BN_Prelu_2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=[3,3],stride_size=[1,1], is_batchnorm=False):
        super(Conv2d_BN_Prelu_2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.PReLU(num_parameters=1,init=0.25))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.PReLU(num_parameters=1,init=0.25))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.PReLU(num_parameters=1,init=0.25))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size,filter_size, stride_size, padding=1),nn.PReLU(num_parameters=1,init=0.25))
                                       
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs






class Conv2d_BN_LeakyReLU_2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=[3,3],stride_size=[1,1], is_batchnorm=False):
        super(Conv2d_BN_LeakyReLU_2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size,filter_size, stride_size, padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True))
                                       
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs





####one time conv2d  BN relu
class Conv2d_BN_Leakyrelu(nn.Module):
    def __init__(self, in_size, out_size,filter_size=[3,3],stride_size=[1,1], is_batchnorm=False):
        super(Conv2d_BN_Leakyrelu, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True))
                                       
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs



####one time conv2d  BN relu
class Conv2d_BN_relu(nn.Module):
    def __init__(self, in_size, out_size,filter_size=[3,3],stride_size=[1,1], is_batchnorm=False):
        super(Conv2d_BN_relu, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.ReLU(inplace=True))
                                       
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs





####two time conv2d  BN relu
class Conv2d_BN_relu_2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=[3,3],stride_size=[1,1], is_batchnorm=False):
        super(Conv2d_BN_relu_2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size,filter_size, stride_size, padding=1),nn.BatchNorm2d(out_size),nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size,filter_size, stride_size, padding=1),nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size,filter_size, stride_size, padding=1),nn.ReLU(inplace=True))
                                       
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs



class torch_cat(nn.Module):
    def __init__(self,):
        super(torch_cat, self).__init__()        
    def forward(self, inputs1, inputs2):        
        return torch.cat([inputs1, inputs2], 1)


class torch_cat_pad(nn.Module):
    def __init__(self,):
        super(torch_cat, self).__init__()
        
    def forward(self, inputs1, inputs2):
        outputs2 = inputs2
        offset1 = (outputs2.size()[2]-inputs1.size()[2])
        offset2 = (outputs2.size()[3]-inputs1.size()[3])
        padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
        # # Skip and concatenate 
        outputs1 = F.pad(inputs1, padding)
        return torch.cat([outputs1, outputs2], 1)

class Vel_net_conv_one(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Vel_net_conv_one, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]
        filters = [32, 64, 128, 128, 128, 128, 128, 128, 64, 32]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_relu(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_relu(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_relu(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.conv3  = Conv2d_BN_relu(filters[2], filters[3], filter_size, stride_size, self.is_batchnorm);
        self.conv4  = Conv2d_BN_relu(filters[3], filters[4], filter_size, stride_size, self.is_batchnorm);
########## step2         
        self.torch_cat  = torch_cat();
########## step3
        self.conv5  = Conv2d_BN_relu(filters[4], filters[5], filter_size, stride_size, self.is_batchnorm);
        self.conv6  = Conv2d_BN_relu(filters[4]+filters[5], filters[6], filter_size, stride_size, self.is_batchnorm);    
        self.conv7  = Conv2d_BN_relu(filters[3]+filters[6], filters[7], filter_size, stride_size, self.is_batchnorm); 
        self.conv8  = Conv2d_BN_relu(filters[2]+filters[7], filters[8], filter_size, stride_size, self.is_batchnorm);
        self.conv9  = Conv2d_BN_relu(filters[1]+filters[8], filters[9], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[0], self.n_classes, [1,1])
   # ,label_dsp_dim     
    # def forward(self, inputs,label_dsp_dim):
    def forward(self, inputs):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        conv3  = self.conv3(conv2)
        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)
        
        c45    = self.torch_cat(conv4,conv5);
        conv6  = self.conv6(c45)
        c36    = self.torch_cat(conv3,conv6);
        conv7  = self.conv7(c36);
        c27    = self.torch_cat(conv2,conv7);
        conv8  = self.conv8(c27);
        c18    = self.torch_cat(conv1,conv8);
        conv9  = self.conv9(c18);
        
        # final    = conv9[:,:,0:0+label_dsp_dim[0],0:0+label_dsp_dim[1]].contiguous()
        final    = conv9
        
        return self.final(final)
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()
                

class Vel_net_conv_one_label_dsp_dim(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Vel_net_conv_one_label_dsp_dim, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]
        # filters = [32, 64, 128, 128, 128, 128, 128, 128, 64, 32]
        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        filters = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_relu(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_relu(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_relu(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.conv3  = Conv2d_BN_relu(filters[2], filters[3], filter_size, stride_size, self.is_batchnorm);
        self.conv4  = Conv2d_BN_relu(filters[3], filters[4], filter_size, stride_size, self.is_batchnorm);
########## step2         
        self.torch_cat  = torch_cat();
########## step3
        self.conv5  = Conv2d_BN_relu(filters[4], filters[5], filter_size, stride_size, self.is_batchnorm);
        self.conv6  = Conv2d_BN_relu(filters[4]+filters[5], filters[6], filter_size, stride_size, self.is_batchnorm);    
        self.conv7  = Conv2d_BN_relu(filters[3]+filters[6], filters[7], filter_size, stride_size, self.is_batchnorm); 
        self.conv8  = Conv2d_BN_relu(filters[2]+filters[7], filters[8], filter_size, stride_size, self.is_batchnorm);
        self.conv9  = Conv2d_BN_relu(filters[1]+filters[8], filters[9], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[0], self.n_classes, [1,1])
   # ,label_dsp_dim     
    # def forward(self, inputs,label_dsp_dim):
    def forward(self, inputs,label_dsp_dim):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        conv3  = self.conv3(conv2)
        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)
        
        c45    = self.torch_cat(conv4,conv5);
        conv6  = self.conv6(c45)
        c36    = self.torch_cat(conv3,conv6);
        conv7  = self.conv7(c36);
        c27    = self.torch_cat(conv2,conv7);
        conv8  = self.conv8(c27);
        c18    = self.torch_cat(conv1,conv8);
        conv9  = self.conv9(c18);
        
        final = self.final(conv9)
                
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        # final    = conv9[:,:,0:0+label_dsp_dim[0],0:0+label_dsp_dim[1]].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()


class Vel_net_conv_one_label_dsp_dim_leaky_relu(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Vel_net_conv_one_label_dsp_dim_leaky_relu, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]
        # filters = [32, 64, 128, 128, 128, 128, 128, 128, 64, 32]
        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        filters = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_Leakyrelu(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_Leakyrelu(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_Leakyrelu(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.conv3  = Conv2d_BN_Leakyrelu(filters[2], filters[3], filter_size, stride_size, self.is_batchnorm);
        self.conv4  = Conv2d_BN_Leakyrelu(filters[3], filters[4], filter_size, stride_size, self.is_batchnorm);
########## step2         
        self.torch_cat  = torch_cat();
########## step3
        self.conv5  = Conv2d_BN_Leakyrelu(filters[4], filters[5], filter_size, stride_size, self.is_batchnorm);
        self.conv6  = Conv2d_BN_Leakyrelu(filters[4]+filters[5], filters[6], filter_size, stride_size, self.is_batchnorm);    
        self.conv7  = Conv2d_BN_Leakyrelu(filters[3]+filters[6], filters[7], filter_size, stride_size, self.is_batchnorm); 
        self.conv8  = Conv2d_BN_Leakyrelu(filters[2]+filters[7], filters[8], filter_size, stride_size, self.is_batchnorm);
        self.conv9  = Conv2d_BN_Leakyrelu(filters[1]+filters[8], filters[9], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[0], self.n_classes, [1,1])
   # ,label_dsp_dim     
    # def forward(self, inputs,label_dsp_dim):
    def forward(self, inputs,label_dsp_dim):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        conv3  = self.conv3(conv2)
        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)
        
        c45    = self.torch_cat(conv4,conv5);
        conv6  = self.conv6(c45)
        c36    = self.torch_cat(conv3,conv6);
        conv7  = self.conv7(c36);
        c27    = self.torch_cat(conv2,conv7);
        conv8  = self.conv8(c27);
        c18    = self.torch_cat(conv1,conv8);
        conv9  = self.conv9(c18);
        
        final = self.final(conv9)
                
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        # final    = conv9[:,:,0:0+label_dsp_dim[0],0:0+label_dsp_dim[1]].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()





                
class  Vel_net_conv_two(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Vel_net_conv_two, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]
        filters = [32, 64, 128, 128, 128, 128, 128, 128, 64, 32]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_relu_2(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_relu_2(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_relu_2(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.conv3  = Conv2d_BN_relu_2(filters[2], filters[3], filter_size, stride_size, self.is_batchnorm);
        self.conv4  = Conv2d_BN_relu_2(filters[3], filters[4], filter_size, stride_size, self.is_batchnorm);
########## step2         
        self.torch_cat  = torch_cat();
########## step3
        self.conv5  = Conv2d_BN_relu_2(filters[4], filters[5], filter_size, stride_size, self.is_batchnorm);
        self.conv6  = Conv2d_BN_relu_2(filters[4]+filters[5], filters[6], filter_size, stride_size, self.is_batchnorm);    
        self.conv7  = Conv2d_BN_relu_2(filters[3]+filters[6], filters[7], filter_size, stride_size, self.is_batchnorm); 
        self.conv8  = Conv2d_BN_relu_2(filters[2]+filters[7], filters[8], filter_size, stride_size, self.is_batchnorm);
        self.conv9  = Conv2d_BN_relu_2(filters[1]+filters[8], filters[9], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[0], self.n_classes, [1,1])
   # ,label_dsp_dim     
    # def forward(self, inputs,label_dsp_dim):
    def forward(self, inputs):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        conv3  = self.conv3(conv2)
        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)
        
        c45    = self.torch_cat(conv4,conv5);
        conv6  = self.conv6(c45)
        c36    = self.torch_cat(conv3,conv6);
        conv7  = self.conv7(c36);
        c27    = self.torch_cat(conv2,conv7);
        conv8  = self.conv8(c27);
        c18    = self.torch_cat(conv1,conv8);
        conv9  = self.conv9(c18);
        
        # final    = conv9[:,:,0:0+label_dsp_dim[0],0:0+label_dsp_dim[1]].contiguous()
        final    = conv9
        
        return self.final(final)
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()                
  
                
class  Vel_net_conv_two_label_dsp_dim(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Vel_net_conv_two_label_dsp_dim, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]
        # filters = [32, 64, 128, 128, 128, 128, 128, 128, 64, 32]
        filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_relu_2(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_relu_2(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_relu_2(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.conv3  = Conv2d_BN_relu_2(filters[2], filters[3], filter_size, stride_size, self.is_batchnorm);
        self.conv4  = Conv2d_BN_relu_2(filters[3], filters[4], filter_size, stride_size, self.is_batchnorm);
########## step2         
        self.torch_cat  = torch_cat();
########## step3
        self.conv5  = Conv2d_BN_relu_2(filters[4], filters[5], filter_size, stride_size, self.is_batchnorm);
        self.conv6  = Conv2d_BN_relu_2(filters[4]+filters[5], filters[6], filter_size, stride_size, self.is_batchnorm);    
        self.conv7  = Conv2d_BN_relu_2(filters[3]+filters[6], filters[7], filter_size, stride_size, self.is_batchnorm); 
        self.conv8  = Conv2d_BN_relu_2(filters[2]+filters[7], filters[8], filter_size, stride_size, self.is_batchnorm);
        self.conv9  = Conv2d_BN_relu_2(filters[1]+filters[8], filters[9], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[0], self.n_classes, [1,1])
   # ,label_dsp_dim     
    def forward(self, inputs,label_dsp_dim):
    # def forward(self, inputs):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        conv3  = self.conv3(conv2)
        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)
        
        c45    = self.torch_cat(conv4,conv5);
        conv6  = self.conv6(c45)
        c36    = self.torch_cat(conv3,conv6);
        conv7  = self.conv7(c36);
        c27    = self.torch_cat(conv2,conv7);
        conv8  = self.conv8(c27);
        c18    = self.torch_cat(conv1,conv8);
        conv9  = self.conv9(c18);
        
        final = self.final(conv9)
        
        # label_dsp_dim1 = inputs.size(2);
        # label_dsp_dim2 = inputs.size(3);
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()  
    
  
    
  
    
class  Vel_net_conv_two_label_dsp_dim_prelu(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Vel_net_conv_two_label_dsp_dim_prelu, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]
        # filters = [32, 64, 128, 128, 128, 128, 128, 128, 64, 32]
        filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_Prelu_2(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_Prelu_2(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_Prelu_2(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.conv3  = Conv2d_BN_Prelu_2(filters[2], filters[3], filter_size, stride_size, self.is_batchnorm);
        self.conv4  = Conv2d_BN_Prelu_2(filters[3], filters[4], filter_size, stride_size, self.is_batchnorm);
########## step2         
        self.torch_cat  = torch_cat();
########## step3
        self.conv5  = Conv2d_BN_Prelu_2(filters[4], filters[5], filter_size, stride_size, self.is_batchnorm);
        self.conv6  = Conv2d_BN_Prelu_2(filters[4]+filters[5], filters[6], filter_size, stride_size, self.is_batchnorm);    
        self.conv7  = Conv2d_BN_Prelu_2(filters[3]+filters[6], filters[7], filter_size, stride_size, self.is_batchnorm); 
        self.conv8  = Conv2d_BN_Prelu_2(filters[2]+filters[7], filters[8], filter_size, stride_size, self.is_batchnorm);
        self.conv9  = Conv2d_BN_Prelu_2(filters[1]+filters[8], filters[9], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[0], self.n_classes, [1,1])
   # ,label_dsp_dim     
    def forward(self, inputs,label_dsp_dim):
    # def forward(self, inputs):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        conv3  = self.conv3(conv2)
        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)
        
        c45    = self.torch_cat(conv4,conv5);
        conv6  = self.conv6(c45)
        c36    = self.torch_cat(conv3,conv6);
        conv7  = self.conv7(c36);
        c27    = self.torch_cat(conv2,conv7);
        conv8  = self.conv8(c27);
        c18    = self.torch_cat(conv1,conv8);
        conv9  = self.conv9(c18);
        
        final = self.final(conv9)
        
        # label_dsp_dim1 = inputs.size(2);
        # label_dsp_dim2 = inputs.size(3);
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()    
    
  
class  Vel_net_conv_two_label_dsp_dim_LeakyReLU(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Vel_net_conv_two_label_dsp_dim_LeakyReLU, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]
        # filters = [32, 64, 128, 128, 128, 128, 128, 128, 64, 32]
        filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_LeakyReLU_2(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_LeakyReLU_2(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_LeakyReLU_2(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.conv3  = Conv2d_BN_LeakyReLU_2(filters[2], filters[3], filter_size, stride_size, self.is_batchnorm);
        self.conv4  = Conv2d_BN_LeakyReLU_2(filters[3], filters[4], filter_size, stride_size, self.is_batchnorm);
########## step2         
        self.torch_cat  = torch_cat();
########## step3
        self.conv5  = Conv2d_BN_LeakyReLU_2(filters[4], filters[5], filter_size, stride_size, self.is_batchnorm);
        self.conv6  = Conv2d_BN_LeakyReLU_2(filters[4]+filters[5], filters[6], filter_size, stride_size, self.is_batchnorm);    
        self.conv7  = Conv2d_BN_LeakyReLU_2(filters[3]+filters[6], filters[7], filter_size, stride_size, self.is_batchnorm); 
        self.conv8  = Conv2d_BN_LeakyReLU_2(filters[2]+filters[7], filters[8], filter_size, stride_size, self.is_batchnorm);
        self.conv9  = Conv2d_BN_LeakyReLU_2(filters[1]+filters[8], filters[9], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[0], self.n_classes, [1,1])
   # ,label_dsp_dim     
    def forward(self, inputs,label_dsp_dim):
    # def forward(self, inputs):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        conv3  = self.conv3(conv2)
        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)
        
        c45    = self.torch_cat(conv4,conv5);
        conv6  = self.conv6(c45)
        c36    = self.torch_cat(conv3,conv6);
        conv7  = self.conv7(c36);
        c27    = self.torch_cat(conv2,conv7);
        conv8  = self.conv8(c27);
        c18    = self.torch_cat(conv1,conv8);
        conv9  = self.conv9(c18);
        
        final = self.final(conv9)
        
        # label_dsp_dim1 = inputs.size(2);
        # label_dsp_dim2 = inputs.size(3);
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()  


class  Vel_net_conv_two_label_dsp_dim_LeakyReLU_no_skip(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Vel_net_conv_two_label_dsp_dim_LeakyReLU_no_skip, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        # filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        # filters = [32, 64, 128, 256, 512, 512, 256, 128, 64, 32]
        # filters = [32, 64, 128, 128, 128, 128, 128, 128, 64, 32]
        filters = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_LeakyReLU_2(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_LeakyReLU_2(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_LeakyReLU_2(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.conv3  = Conv2d_BN_LeakyReLU_2(filters[2], filters[3], filter_size, stride_size, self.is_batchnorm);
        self.conv4  = Conv2d_BN_LeakyReLU_2(filters[3], filters[4], filter_size, stride_size, self.is_batchnorm);
########## step2         
        self.torch_cat  = torch_cat();
########## step3
        self.conv5  = Conv2d_BN_LeakyReLU_2(filters[4], filters[5], filter_size, stride_size, self.is_batchnorm);
        self.conv6  = Conv2d_BN_LeakyReLU_2(filters[4], filters[6], filter_size, stride_size, self.is_batchnorm);    
        self.conv7  = Conv2d_BN_LeakyReLU_2(filters[3], filters[7], filter_size, stride_size, self.is_batchnorm); 
        self.conv8  = Conv2d_BN_LeakyReLU_2(filters[2], filters[8], filter_size, stride_size, self.is_batchnorm);
        self.conv9  = Conv2d_BN_LeakyReLU_2(filters[1], filters[9], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[0], self.n_classes, [1,1])
   # ,label_dsp_dim     
    def forward(self, inputs,label_dsp_dim):
    # def forward(self, inputs):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        conv3  = self.conv3(conv2)
        conv4  = self.conv4(conv3)
        conv5  = self.conv5(conv4)
        
        # c45    = self.torch_cat(conv4,conv5);
        conv6  = self.conv6(conv5)
        # c36    = self.torch_cat(conv3,conv6);
        conv7  = self.conv7(conv6);
        # c27    = self.torch_cat(conv2,conv7);
        conv8  = self.conv8(conv7);
        # c18    = self.torch_cat(conv1,conv8);
        conv9  = self.conv9(conv8);
        
        final = self.final(conv9)
        
        # label_dsp_dim1 = inputs.size(2);
        # label_dsp_dim2 = inputs.size(3);
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()





    
  
###three   conv1    ###three   conv1        
class  Iter_net_one(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Iter_net_one, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        filters = [64, 64, 64]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_relu(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_relu(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_relu(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[2], self.n_classes, [1,1])
        
    def forward(self, inputs):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        
        return self.final(conv2)
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()                 
                
                
                
                
                
                
                
                
                
###three   conv2        ###three   conv2             
class  Iter_net_two(nn.Module):
    def __init__(self, n_classes, in_channels , is_batchnorm):
        super(Iter_net_two, self).__init__()
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes

        filters = [64, 64, 64]
        filter_size=[3,3]
        stride_size=[1,1]
        
########## step1       
        self.conv0  = Conv2d_BN_relu_2(self.in_channels, filters[0], filter_size, stride_size, self.is_batchnorm);
        self.conv1  = Conv2d_BN_relu_2(filters[0], filters[1], filter_size, stride_size, self.is_batchnorm);
        self.conv2  = Conv2d_BN_relu_2(filters[1], filters[2], filter_size, stride_size, self.is_batchnorm);
        self.final   = nn.Conv2d(filters[2], self.n_classes, [1,1])
        
    def forward(self, inputs):
        conv0  = self.conv0(inputs)
        conv1  = self.conv1(conv0)
        conv2  = self.conv2(conv1)
        
        return self.final(conv2)
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()                                 
  
                
  
    
  
    
  
    
  
    
######################            
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################          
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
###################### 
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################           
######################
##nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
############ 3*3 conv   +   BN  +  relu
class conv2relu(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(conv2relu, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
            
    def forward(self, inputs):
        conv1_result = self.conv1(inputs)
        return conv1_result



class conv2relu_conv2relu(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(conv2relu_conv2relu, self).__init__()
        # Kernel size: 3*3, Stride: 1, Padding: 1
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs



class conv2relu_conv2relu_down(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(conv2relu_conv2relu_down, self).__init__()
        self.conv = conv2relu_conv2relu(in_size, out_size, is_batchnorm)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        conv_result = self.conv(inputs)
        maxpool_result = self.down(conv_result)
        return conv_result,maxpool_result



class upconv_concat(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, is_deconv):
        super(upconv_concat, self).__init__()
        self.is_batchnorm = is_batchnorm;
        self.is_deconv    = is_deconv;
         
        # Transposed convolution
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,stride=2)
        
        # interpolation and 3*3 convolution 
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.conv = conv2relu(in_size, out_size, is_batchnorm);
            if is_batchnorm:
                self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
            else:
                self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.ReLU(inplace=True),)

    def forward(self, inputs1, inputs2):
        # print("inputs1 is {}".format(inputs1.size()));
        # print("inputs2 is {}".format(inputs2.size()));

        outputs2 = self.up(inputs2)
        
        if not self.is_deconv:
            outputs2 = self.conv(outputs2);
       
        
       
        # Skip and concatenate     
        offset1 = (outputs2.size()[2]-inputs1.size()[2])
        offset2 = (outputs2.size()[3]-inputs1.size()[3])
        # padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
        # outputs1 = F.pad(inputs1, padding)
        # print("offset1 is {}".format(offset1));
        # print("offset2 is {}".format(offset2));
     
        Pad = nn.ReplicationPad2d(padding=(0,offset2,0,offset1));##left right top bottom
        outputs1  = Pad(inputs1);
        
        # print("outputs1 is {}".format(outputs1.size()));
        # print("outputs2 is {}".format(outputs2.size()));
        return torch.cat([outputs1, outputs2], 1)














class  Vel_unet(nn.Module):
    def __init__(self, n_classes, in_channels ,is_deconv=True, is_batchnorm=True):
        super(Vel_unet, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes
        
        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        # filters = [32, 64, 128, 128, 128]
        
        self.conv1   = conv2relu_conv2relu_down(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2   = conv2relu_conv2relu_down(filters[0], filters[1], self.is_batchnorm)
        self.conv3   = conv2relu_conv2relu_down(filters[1], filters[2], self.is_batchnorm)
        self.conv4   = conv2relu_conv2relu_down(filters[2], filters[3], self.is_batchnorm)
        self.conv5   = conv2relu_conv2relu(filters[3], filters[4], self.is_batchnorm)
        
        self.up6     = upconv_concat(filters[4], filters[3], self.is_batchnorm, self.is_deconv)
        self.conv6   = conv2relu_conv2relu(filters[4], filters[3], self.is_batchnorm)
        
        self.up7     = upconv_concat(filters[3], filters[2], self.is_batchnorm, self.is_deconv)
        self.conv7   = conv2relu_conv2relu(filters[3], filters[2], self.is_batchnorm)
        
        self.up8     = upconv_concat(filters[2], filters[1], self.is_batchnorm, self.is_deconv)
        self.conv8   = conv2relu_conv2relu(filters[2], filters[1], self.is_batchnorm)
        
        self.up9     = upconv_concat(filters[1], filters[0], self.is_batchnorm, self.is_deconv)
        self.conv9   = conv2relu_conv2relu(filters[1], filters[0], self.is_batchnorm)
        
        self.final   = nn.Conv2d(filters[0],self.n_classes, 1)
        
    def forward(self, inputs):
        conv1, down1  = self.conv1(inputs);#print("conv1 {} down1 {}".format(conv1.size(),down1.size()));
        conv2, down2  = self.conv2(down1);#print("conv2 {} down2 {}".format(conv2.size(),down2.size()));
        conv3, down3  = self.conv3(down2);#print("conv3 {} down3 {}".format(conv3.size(),down3.size()));
        conv4, down4  = self.conv4(down3);#print("conv4 {} down4 {}".format(conv4.size(),down4.size()));
        conv5         = self.conv5(down4);    #print("conv5 is {}".format(conv5.size()));
        
        up6           = self.up6(conv4, conv5);#print("up6 is {}".format(up6.size()));
        conv6         = self.conv6(up6);       #print("conv6 is {}".format(conv6.size()));
        
        up7           = self.up7(conv3, conv6);#print("up7 is {}".format(up7.size()));
        conv7         = self.conv7(up7);       #print("conv7 is {}".format(conv7.size()));
        
        up8           = self.up8(conv2, conv7);#print("up8 is {}".format(up8.size()));
        conv8         = self.conv8(up8);       #print("conv8 is {}".format(conv8.size()));
        
        up9           = self.up9(conv1, conv8);#print("up9 is {}".format(up9.size()));
        conv9         = self.conv9(up9);       #print("conv9 is {}".format(conv9.size()));
        
        final         = self.final(conv9);
        
        label_dsp_dim1 = inputs.size(2);
        label_dsp_dim2 = inputs.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        # up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_() 
                    
# net = Vel_unet(n_classes=Nclasses,in_channels=Inchannels,is_deconv=True,is_batchnorm=False)
# down1_size is torch.Size([10, 32, 100, 90])
# down2_size is torch.Size([10, 64, 50, 45])
# down3_size is torch.Size([10, 128, 25, 23])
# down4_size is torch.Size([10, 256, 13, 12])

# center_size is torch.Size([10, 512, 13, 12])

# outputs1 is torch.Size([10, 256, 26, 24])
# outputs2 is torch.Size([10, 256, 26, 24])
# up4 is torch.Size([10, 256, 26, 24])

# outputs1 is torch.Size([10, 128, 52, 48])
# outputs2 is torch.Size([10, 128, 52, 48])
# up3 is torch.Size([10, 128, 52, 48])

# outputs1 is torch.Size([10, 64, 104, 96])
# outputs2 is torch.Size([10, 64, 104, 96])
# up2 is torch.Size([10, 64, 104, 96])

# outputs1 is torch.Size([10, 32, 208, 192])
# outputs2 is torch.Size([10, 32, 208, 192])
# up1 is torch.Size([10, 32, 208, 192])

# net_outputs shapetorch.Size([10, 1, 208, 192])


# net = Vel_unet(n_classes=Nclasses,in_channels=Inchannels,is_deconv=False,is_batchnorm=False)
# down1_size is torch.Size([10, 32, 100, 90])
# down2_size is torch.Size([10, 64, 50, 45])
# down3_size is torch.Size([10, 128, 25, 23])
# down4_size is torch.Size([10, 256, 13, 12])
# center_size is torch.Size([10, 512, 13, 12])

# outputs1 is torch.Size([10, 256, 26, 24])
# outputs2 is torch.Size([10, 256, 26, 24])
# up4 is torch.Size([10, 256, 26, 24])

# outputs1 is torch.Size([10, 128, 52, 48])
# outputs2 is torch.Size([10, 128, 52, 48])
# up3 is torch.Size([10, 128, 52, 48])

# outputs1 is torch.Size([10, 64, 104, 96])
# outputs2 is torch.Size([10, 64, 104, 96])
# up2 is torch.Size([10, 64, 104, 96])

# outputs1 is torch.Size([10, 32, 208, 192])
# outputs2 is torch.Size([10, 32, 208, 192])
# up1 is torch.Size([10, 32, 208, 192])

# net_outputs shapetorch.Size([10, 1, 208, 192])



class  Vel_unet_label_dsp_dim(nn.Module):
    def __init__(self, n_classes, in_channels ,is_deconv, is_batchnorm,initial_filter_num):
        super(Vel_unet_label_dsp_dim, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes
        self.initial_filter_num     =initial_filter_num
        
        # filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]
        filters = [initial_filter_num, initial_filter_num*2, initial_filter_num*4, initial_filter_num*8, initial_filter_num*16]
        # filters = [32, 64, 128, 128, 128]
        
        self.conv1   = conv2relu_conv2relu_down(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2   = conv2relu_conv2relu_down(filters[0], filters[1], self.is_batchnorm)
        self.conv3   = conv2relu_conv2relu_down(filters[1], filters[2], self.is_batchnorm)
        self.conv4   = conv2relu_conv2relu_down(filters[2], filters[3], self.is_batchnorm)
        self.conv5   = conv2relu_conv2relu(filters[3], filters[4], self.is_batchnorm)
        
        self.up6     = upconv_concat(filters[4], filters[3], self.is_batchnorm, self.is_deconv)
        self.conv6   = conv2relu_conv2relu(filters[4], filters[3], self.is_batchnorm)
        
        self.up7     = upconv_concat(filters[3], filters[2], self.is_batchnorm, self.is_deconv)
        self.conv7   = conv2relu_conv2relu(filters[3], filters[2], self.is_batchnorm)
        
        self.up8     = upconv_concat(filters[2], filters[1], self.is_batchnorm, self.is_deconv)
        self.conv8   = conv2relu_conv2relu(filters[2], filters[1], self.is_batchnorm)
        
        self.up9     = upconv_concat(filters[1], filters[0], self.is_batchnorm, self.is_deconv)
        self.conv9   = conv2relu_conv2relu(filters[1], filters[0], self.is_batchnorm)
        
        self.final   = nn.Conv2d(filters[0],self.n_classes, 1)
        
    def forward(self, inputs,label_dsp_dim):
        conv1, down1  = self.conv1(inputs);#print("conv1 {} down1 {}".format(conv1.size(),down1.size()));
        conv2, down2  = self.conv2(down1);#print("conv2 {} down2 {}".format(conv2.size(),down2.size()));
        conv3, down3  = self.conv3(down2);#print("conv3 {} down3 {}".format(conv3.size(),down3.size()));
        conv4, down4  = self.conv4(down3);#print("conv4 {} down4 {}".format(conv4.size(),down4.size()));
        conv5         = self.conv5(down4);    #print("conv5 is {}".format(conv5.size()));
        
        up6           = self.up6(conv4, conv5);#print("up6 is {}".format(up6.size()));
        conv6         = self.conv6(up6);       #print("conv6 is {}".format(conv6.size()));
        
        up7           = self.up7(conv3, conv6);#print("up7 is {}".format(up7.size()));
        conv7         = self.conv7(up7);       #print("conv7 is {}".format(conv7.size()));
        
        up8           = self.up8(conv2, conv7);#print("up8 is {}".format(up8.size()));
        conv8         = self.conv8(up8);       #print("conv8 is {}".format(conv8.size()));
        
        up9           = self.up9(conv1, conv8);#print("up9 is {}".format(up9.size()));
        conv9         = self.conv9(up9);       #print("conv9 is {}".format(conv9.size()));
        
        final         = self.final(conv9);
        
        # label_dsp_dim1 = inputs.size(2);
        # label_dsp_dim2 = inputs.size(3);
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        # up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_() 
                    
############
###########                    
############
###########     
############
###########     
############
###########     
############
###########     
############
###########     
############
###########     
############
###########     
############
###########     
############Wu and McMechan
###########
###########     
############
###########     
############
###########     
############
###########     
############Wu and McMechan
###########
class upconv_LeakyReLU_1(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=3,stride=1):
        super(upconv_LeakyReLU_1, self).__init__()
       
        if is_batchnorm:
            self.up = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        else:
            self.up = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        
        
    def forward(self, inputs1):
        outputs1 = self.up(inputs1);
       
        return outputs1


class upconv_LeakyReLU_4(nn.Module):
    def __init__(self, in_size, out_size,is_batchnorm, kernel_size=3,stride=1):
        super(upconv_LeakyReLU_4, self).__init__()
        if is_batchnorm:
            self.up1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up2 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));  
            self.up3 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up4 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        else:
            self.up1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up2 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));  
            self.up3 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up4 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        
    def forward(self, inputs1):
        outputs1 = self.up1(inputs1);
        outputs1 = self.up2(outputs1);
        outputs1 = self.up3(outputs1);
        outputs1 = self.up4(outputs1);
       
        return outputs1
    
    
class upconv_LeakyReLU_5(nn.Module):
    def __init__(self, in_size, out_size,is_batchnorm, kernel_size=3,stride=2):
        super(upconv_LeakyReLU_5, self).__init__()
        if is_batchnorm:
            self.up1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up2 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));  
            self.up3 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up4 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up5 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        else:
            self.up1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up2 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));  
            self.up3 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up4 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up5 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        
    def forward(self, inputs1):
        outputs1 = self.up1(inputs1);
        outputs1 = self.up2(outputs1);
        outputs1 = self.up3(outputs1);
        outputs1 = self.up4(outputs1);
        outputs1 = self.up5(outputs1);
       
        return outputs1   
    

    
class upconv(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=2):
        super(upconv, self).__init__()
       
        self.up = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride));
        
    def forward(self, inputs1):
        outputs1 = self.up(inputs1);
       
        return outputs1

class upconv_tanh(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3,stride=2):
        super(upconv_tanh, self).__init__()
       
        self.up = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,stride=stride),nn.Tanh());
        
    def forward(self, inputs1):
        outputs1 = self.up(inputs1);
       
        return outputs1

                         
class  Wu_and_McMechan_net_label_dsp_dim(nn.Module):
    def __init__(self, n_classes, in_channels ,is_deconv, is_batchnorm):
        super(Wu_and_McMechan_net_label_dsp_dim, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes
        
        # filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]
        # filters = [32, 64, 128, 128, 128]
        filters = [32, 32, 32, 32, 32]
        
        self.up1   = upconv_LeakyReLU_1(self.in_channels, filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        self.up2   = upconv_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        
        self.up3   = upconv_LeakyReLU_1(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=2);
        self.up4   = upconv_LeakyReLU_4(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        
        self.up5   = upconv_LeakyReLU_1(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=2);
        self.up6   = upconv_LeakyReLU_4(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        

        self.up7   = upconv_LeakyReLU_1(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=2);
        self.up8   = upconv_LeakyReLU_4(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        
        self.up9   = upconv_LeakyReLU_1(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=2);
        self.up10   = upconv_LeakyReLU_4(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);


        self.final   = upconv(filters[0],self.n_classes, kernel_size=3,stride=2);
        # self.final   = upconv_tanh(filters[0],self.n_classes, kernel_size=3,stride=2);
        
        
    def forward(self, inputs,label_dsp_dim):
        
        up1 =   self.up1(inputs);   print(" up1 {}".format( up1.size()));
        up2 =   self.up2(up1);      print(" up2 {}".format( up2.size()));
        up3 =   self.up3(up2);      print(" up3 {}".format( up3.size()));
        up4 =   self.up4(up3);      print(" up4 {}".format( up4.size()));
        up5 =   self.up5(up4);      print(" up5 {}".format( up5.size()));
        up6 =   self.up6(up5);      print(" up6 {}".format( up6.size()));
        up7 =   self.up7(up6);      print(" up7 {}".format( up7.size()));
        up8 =   self.up8(up7);      print(" up8 {}".format( up8.size()));
        up9 =   self.up9(up8);      print(" up9 {}".format( up9.size()));
        up10 =  self.up10(up9);     print(" up10 {}".format(up10.size()));

        final         = self.final(up10);   print(" up10 {}".format(final.size()));
        
        # label_dsp_dim1 = inputs.size(2);
        # label_dsp_dim2 = inputs.size(3);
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        # up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()                     
                    
                    
                    
############
###########                    
############
###########     
############
###########     
############
###########     
############
###########     
############
###########     
############
###########     
############
###########     
############
###########     
############Zhang and 
###########
###########     
############
###########     
############
###########     
############
###########     
############
###########
# class upconv_concat(nn.Module):
#     def __init__(self, in_size, out_size, is_batchnorm, is_deconv):
#         super(upconv_concat, self).__init__()
#         self.is_batchnorm = is_batchnorm;
#         self.is_deconv    = is_deconv;
         
#         # Transposed convolution
#         if is_deconv:
#             self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,stride=2)
        
#         # interpolation and 3*3 convolution 
#         else:
#             self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#             # self.conv = conv2relu(in_size, out_size, is_batchnorm);
#             if is_batchnorm:
#                 self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
#                                        nn.BatchNorm2d(out_size),
#                                        nn.ReLU(inplace=True),)
#             else:
#                 self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
#                                        nn.ReLU(inplace=True),)

#     def forward(self, inputs1, inputs2):
#         # print("inputs1 is {}".format(inputs1.size()));
#         # print("inputs2 is {}".format(inputs2.size()));

#         outputs2 = self.up(inputs2)
        
#         if not self.is_deconv:
#             outputs2 = self.conv(outputs2);
       
        
       
#         # Skip and concatenate     
#         offset1 = (outputs2.size()[2]-inputs1.size()[2])
#         offset2 = (outputs2.size()[3]-inputs1.size()[3])
#         # padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
#         # outputs1 = F.pad(inputs1, padding)
#         # print("offset1 is {}".format(offset1));
#         # print("offset2 is {}".format(offset2));
     
#         Pad = nn.ReplicationPad2d(padding=(0,offset2,0,offset1));##left right top bottom
#         outputs1  = Pad(inputs1);
        
#         # print("outputs1 is {}".format(outputs1.size()));
#         # print("outputs2 is {}".format(outputs2.size()));
#         return torch.cat([outputs1, outputs2], 1)

class interpolation_size(nn.Module):
    def __init__(self, scale_factor=2):
        super(interpolation_size, self).__init__()
       
        self.up = nn.UpsamplingBilinear2d(scale_factor=2);
                
    def forward(self, inputs1):
        outputs1 = self.up(inputs1);
       
        return outputs1

class interpolation_LeakyReLU_5(nn.Module):
    def __init__(self, in_size, out_size,is_batchnorm, kernel_size=3,stride=2):
        super(interpolation_LeakyReLU_5, self).__init__()
        
        self.interpolation = interpolation_size(scale_factor=2);
        
        if is_batchnorm:
            self.up1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));  
            self.up3 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up4 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up5 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        else:
            self.up1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True));  
            self.up3 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up4 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
            self.up5 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        
    def forward(self, inputs1):
        
        tmp = self.interpolation(inputs1);  #print(" tmp  {}".format( tmp.size()));
        
        outputs1 = self.up1(tmp);           #print("outputs1  {}".format( outputs1.size()));
        outputs1 = self.up2(outputs1);      #print("outputs1  {}".format( outputs1.size()));
        outputs1 = self.up3(outputs1);
        outputs1 = self.up4(outputs1);
        outputs1 = self.up5(outputs1);      #print("outputs1  {}".format( outputs1.size()));
       
        return outputs1
    
class LeakyReLU_1(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=3,stride=1):
        super(LeakyReLU_1, self).__init__()
       
        if is_batchnorm:
            self.up = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.BatchNorm2d(out_size),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        else:
            self.up = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=kernel_size,stride=stride,padding=1),nn.LeakyReLU(negative_slope = 0.2, inplace=True));
        
        
    def forward(self, inputs1):
        outputs1 = self.up(inputs1);
       
        return outputs1    


class  Wu_and_McMechan_net_label_dsp_dim_new(nn.Module):
    def __init__(self, n_classes, in_channels ,is_deconv, is_batchnorm):
        super(Wu_and_McMechan_net_label_dsp_dim_new, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes
        
        # filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]
        # filters = [32, 64, 128, 128, 128]
        
        filters = [32, 32, 32, 32, 32]
        
        self.conv1 = LeakyReLU_1(self.in_channels, filters[0],self.is_batchnorm, kernel_size=1,stride=1);
        
        self.up1   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        
        self.up2   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
    
        self.up3   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        self.up4   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        
        self.up5   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        self.up6   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
       
        self.final   = upconv(filters[0],self.n_classes, kernel_size=3,stride=1);
        # self.final   = upconv_tanh(filters[0],self.n_classes, kernel_size=3,stride=1);
        
        
    def forward(self, inputs,label_dsp_dim):
        
        conv1 = self.conv1(inputs);   print(" conv1 {}".format( conv1.size()));
        
        up1 =   self.up1(conv1);   print(" up1 {}".format( up1.size()));
        up2 =   self.up2(up1);      print(" up2 {}".format( up2.size()));
        up3 =   self.up3(up2);      print(" up3 {}".format( up3.size()));
        up4 =   self.up4(up3);      print(" up4 {}".format( up4.size()));
        up5 =   self.up5(up4);      print(" up5 {}".format( up5.size()));
        up6 =   self.up6(up5);      print(" up6 {}".format( up6.size()));

        final         = self.final(up6);   print(" up10 {}".format(final.size()));
        
        # label_dsp_dim1 = inputs.size(2);
        # label_dsp_dim2 = inputs.size(3);
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        # up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_() 

    

class  Wu_and_McMechan_net_label_dsp_dim_new_new(nn.Module):
    def __init__(self, n_classes, in_channels ,is_deconv, is_batchnorm):
        super(Wu_and_McMechan_net_label_dsp_dim_new_new, self).__init__()
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     =n_classes
        
        # filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]
        # filters = [32, 64, 128, 128, 128]
        
        filters = [32, 32, 32, 32, 32]
        
        self.interpolation = interpolation_size(scale_factor=2);
        
        self.conv1 = LeakyReLU_1(self.in_channels, filters[0],self.is_batchnorm, kernel_size=1,stride=1);
        
        self.up1   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        
        self.up2   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
    
        self.up3   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        self.up4   = interpolation_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        
        self.up5   = interpolation_LeakyReLU_5(filters[0]*2, filters[0],self.is_batchnorm, kernel_size=3,stride=1);
        
        self.up6   = interpolation_LeakyReLU_5(filters[0]*2, filters[0],self.is_batchnorm, kernel_size=3,stride=1);
       
        self.final   = upconv(filters[0],self.n_classes, kernel_size=3,stride=1);
        # self.final   = upconv_tanh(filters[0],self.n_classes, kernel_size=3,stride=1);
        
        
    def forward(self, inputs,label_dsp_dim):
        
        conv1 = self.conv1(inputs);         print(" conv1 {}".format( conv1.size()));
        
        up1 =   self.up1(conv1);            print(" up1 {}".format( up1.size()));
        up2 =   self.up2(up1);              print(" up2 {}".format( up2.size()));
        up3 =   self.up3(up2);              print(" up3 {}".format( up3.size()));
        up4 =   self.up4(up3);              print(" up4 {}".format( up4.size()));
        
        
        uu1 = self.interpolation(up3);      print("  uu1 {}".format(  uu1.size()));
        conta1 = torch.cat([up4,uu1], 1);   print(" conta1 {}".format( conta1.size()));
        
        
        
        up5 =   self.up5(conta1);           print(" up5 {}".format( up5.size()));
        
        
        uu1 = self.interpolation(up2)
        uu2 = self.interpolation(uu1)
        uu3 = self.interpolation(uu2)
        conta2 = torch.cat([up5,uu3], 1) ;  print(" conta2 {}".format(conta2.size()));
        
        
        
        up6 =   self.up6(conta2);           print(" up6 {}".format( up6.size()));
        
        
        
     

        final         = self.final(up6);   print(" up10 {}".format(final.size()));
        
        # label_dsp_dim1 = inputs.size(2);
        # label_dsp_dim2 = inputs.size(3);
        label_dsp_dim1 = label_dsp_dim[0]
        label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        # up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def  _initialize_weights(self):
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                # m.weight.data.zero_()
                m.bias.data.zero_()
            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, torch.sqrt(2. / n))
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_() 




