#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 00:56:14 2023

@author: zhangjiwei
"""
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
# import plot_func as P  #from func import plot_func as P

# Python2 : super(conv2relu_conv2relu, self).__init__()
# Python3 : super().__init__()

# CLASStorch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels (int) – Number of channels in the input image
# out_channels (int) – Number of channels produced by the convolution
# kernel_size (int or tuple) – Size of the convolving kernel
# stride (int or tuple, optional) – Stride of the convolution. Default: 1
# padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
# padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
# dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
# groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1

# bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
# padding='valid' is the same as no padding. padding='same' pads the input so the output has the shape as the input. However, this mode doesn’t support any stride values other than 1.


#in_size, out_size, kernel_size, padding, padding_mode, is_deconv, is_batchnorm, spectral_norm, bias_bool1, Leaky

#self.conv = conv2relu_conv2relu(in_size, out_size, kernel_size = kernel_size, padding = padding, padding_mode = padding_mode, is_deconv = is_deconv, is_batchnorm = is_batchnorm, spectral_norm = spectral_norm, bias_bool1 = bias_bool1, Leaky = Leaky);
# self.conv5   = conv2relu_conv2relu(filters[3], filters[4], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
class conv2relu_conv2relu(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=True, is_batchnorm=True, spectral_norm=True, bias_bool1=True, Leaky=0 ):
        super().__init__()
        
        if is_batchnorm and spectral_norm:
            self.conv1 = nn.Sequential( nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_size, out_size, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) ),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(negative_slope = Leaky), )
            self.conv2 = nn.Sequential( nn.utils.parametrizations.spectral_norm( nn.Conv2d(out_size, out_size, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) ),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(negative_slope = Leaky), )


        if is_batchnorm and not spectral_norm:
            self.conv1 = nn.Sequential(  nn.Conv2d(in_size, out_size, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1),
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(negative_slope = Leaky), )
            self.conv2 = nn.Sequential(  nn.Conv2d(out_size, out_size, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) ,
                                       nn.BatchNorm2d(out_size),
                                       nn.LeakyReLU(negative_slope = Leaky), )
 
        
        if not is_batchnorm and spectral_norm:
            self.conv1 = nn.Sequential( nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_size, out_size, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) ),
                                       
                                       nn.LeakyReLU(negative_slope = Leaky), )
            self.conv2 = nn.Sequential( nn.utils.parametrizations.spectral_norm( nn.Conv2d(out_size, out_size, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) ),
                                       
                                       nn.LeakyReLU(negative_slope = Leaky), )


        if not is_batchnorm and not spectral_norm:
            self.conv1 = nn.Sequential(  nn.Conv2d(in_size, out_size, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1),
                                       
                                       nn.LeakyReLU(negative_slope = Leaky), )
            self.conv2 = nn.Sequential(  nn.Conv2d(out_size, out_size, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) ,
                                       
                                       nn.LeakyReLU(negative_slope = Leaky), )
            
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


#conv2relu_conv2relu_down(self.in_channels, filters[0], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
class conv2relu_conv2relu_down(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=True, is_batchnorm=True, spectral_norm=True, bias_bool1=True, Leaky=0 ):
        super().__init__()
        
        self.conv = conv2relu_conv2relu(in_size, out_size, kernel_size = kernel_size, padding = padding, padding_mode = padding_mode, is_deconv = is_deconv, is_batchnorm = is_batchnorm, spectral_norm = spectral_norm, bias_bool1 = bias_bool1, Leaky = Leaky);
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)
        
    def forward(self, inputs):
        conv_result = self.conv( inputs )
        maxpool_result = self.down(conv_result)
        return conv_result,maxpool_result


class upconv_concat(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=True, is_batchnorm=True, spectral_norm=True, bias_bool1=True, Leaky=0 ):
        super().__init__()
        
        self.is_batchnorm = is_batchnorm;
        self.is_deconv    = is_deconv;
         
        # Transposed convolution
        if self.is_deconv:
            if spectral_norm:
                self.up = nn.utils.parametrizations.spectral_norm(   nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=bias_bool1)  )
            if not spectral_norm:
                self.up =  nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=bias_bool1)
        
        
        # interpolation and 3*3 convolution 
        else: 
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

            if is_batchnorm and spectral_norm:
                self.conv1 = nn.Sequential( nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, padding_mode='replicate', bias=bias_bool1) ),
                                           nn.BatchNorm2d(out_size),
                                           nn.LeakyReLU(negative_slope = Leaky), )

            if is_batchnorm and not spectral_norm:
                self.conv1 = nn.Sequential( nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, padding_mode='replicate', bias=bias_bool1) ,
                                           nn.BatchNorm2d(out_size),
                                           nn.LeakyReLU(negative_slope = Leaky), )
            
            if not is_batchnorm and spectral_norm:
                self.conv1 = nn.Sequential( nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, padding_mode='replicate', bias=bias_bool1) ),

                                           nn.LeakyReLU(negative_slope = Leaky), )

            if not is_batchnorm and not spectral_norm:
                self.conv1 = nn.Sequential( nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, padding_mode='replicate', bias=bias_bool1) ,

                                           nn.LeakyReLU(negative_slope = Leaky), )

    def forward(self, inputs1, inputs2):
        # print("inputs1 is {}".format(inputs1.size()));
        # print("inputs2 is {}".format(inputs2.size()));

        outputs2 = self.up(inputs2)
        
        if not self.is_deconv:
            outputs2 = self.conv1(outputs2);
       
        # Skip and concatenate
        offset1 = (outputs2.size()[2]-inputs1.size()[2])
        offset2 = (outputs2.size()[3]-inputs1.size()[3])
        # padding=[offset2//2,(offset2+1)//2,offset1//2,(offset1+1)//2]
        # outputs1 = F.pad(inputs1, padding)
        # print("offset1 is {}".format(offset1));
        # print("offset2 is {}".format(offset2));
     
        Pad = nn.ReplicationPad2d(padding=(0,offset2,0,offset1)); ##left right top bottom
        outputs1  = Pad(inputs1);
        
        # print("outputs1 is {}".format(outputs1.size()));
        # print("outputs2 is {}".format(outputs2.size()));
        return torch.cat([outputs1, outputs2], 1)



# =============================================================================
#DEM_net_1,unet
# =============================================================================
class  DEM_net_1(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        
        self.in_channels    = in_channels
        self.kernel_size    = kernel_size
        self.padding        = padding
        self.padding_mode   = padding_mode
        
        self.is_deconv      = is_deconv
        self.is_batchnorm   = is_batchnorm
        self.spectral_norm  = spectral_norm
        self.init_weight    = init_weight
        self.bias_bool1     = bias_bool1
        self.bias_bool2     = bias_bool2
        self.Leaky          = Leaky
        self.drop_layer     = drop_layer
        
        self.n_classes      = ou_channels
        self.filter_num     = filter_num
        
        filters = [self.filter_num, 2*self.filter_num, 4*self.filter_num, 8*self.filter_num, 16*self.filter_num]
        
        self.conv1   = conv2relu_conv2relu_down(self.in_channels, filters[0], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        self.conv2   = conv2relu_conv2relu_down(filters[0], filters[1], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        self.conv3   = conv2relu_conv2relu_down(filters[1], filters[2], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        self.conv4   = conv2relu_conv2relu_down(filters[2], filters[3], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        self.conv5   = conv2relu_conv2relu(filters[3], filters[4], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        
        
        self.up6     = upconv_concat(filters[4], filters[3], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        self.conv6   = conv2relu_conv2relu(filters[4], filters[3], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        
        self.up7     = upconv_concat(filters[3], filters[2], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        self.conv7   = conv2relu_conv2relu(filters[3], filters[2], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        
        self.up8     = upconv_concat(filters[2], filters[1], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        self.conv8   = conv2relu_conv2relu(filters[2], filters[1], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        
        self.up9     = upconv_concat(filters[1], filters[0], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        self.conv9   = conv2relu_conv2relu(filters[1], filters[0], self.kernel_size, self.padding, self.padding_mode, self.is_deconv, self.is_batchnorm, self.spectral_norm, self.bias_bool1, self.Leaky)
        
        self.final   = nn.Conv2d(filters[0],self.n_classes, kernel_size=1, bias=self.bias_bool2)

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
        # print("DEM_net_1 final.shape {}".format(final.shape));
        # print("DEM_net_1 final_clip.shape {}".format(final_clip.shape));
        return final_clip
    
    # Initialization of Parameters
    # def  _initialize_weights(self):
    #     for m in self.modules():
            
    #         print("m is ".format(m));
    #         print("m is ".format(m));
            
    #         if isinstance(m, nn.Conv2d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, torch.sqrt(2. / n))
    #             # m.weight.data.zero_()
    #             m.weight.data.normal_(0, self.init_weight)
                
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
            
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             # m.weight.data.zero_()
    #             m.bias.data.zero_()
                
    #         if isinstance(m,nn.ConvTranspose2d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, torch.sqrt(2. / n))
    #             # m.weight.data.zero_()
    #             m.weight.data.normal_(0, self.init_weight)
                
    #             if m.bias is not None:
    #                 m.bias.data.zero_()


def write_txt(file_name,output_some,type='w+'):
    data=open(file_name,type);##'w+' 'a+'
    print(output_some,file=data)
    data.close()

def Initialization_net(model, init_weight, file_name):
    
    count = 1;
    file = "model is {} \n\n\n\n\n\n\n\n".format(model)
    write_txt(file_name, file, type='w+');
    
    for m in model.modules():
        count = count + 1;
        
        write_txt(file_name, "Model " + str(count) + " is :" + str(m), type='a+');
        write_txt(file_name, "count is " + str(count), type='a+');
        
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, init_weight)
            write_txt(file_name, str(m), type='a+');
            write_txt(file_name, "nn.Conv2d.weight initial", type='a+');
                    
            if m.bias is not None: 
                m.bias.data.zero_()
                write_txt(file_name, "nn.Conv2d.bias initial", type='a+');
        
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
            write_txt(file_name, str(m), type='a+');
            write_txt(file_name, "nn.BatchNorm2d initial", type='a+');
                    
        if isinstance(m,nn.ConvTranspose2d):
            m.weight.data.normal_(0, init_weight)
            write_txt(file_name, str(m), type='a+');
            write_txt(file_name, "nn.ConvTranspose2d.weight initial", type='a+');
            if m.bias is not None:
                m.bias.data.zero_()
                write_txt(file_name, "nn.ConvTranspose2d.bias initial", type='a+');
        
        write_txt(file_name, "\n\n\n\n", type='a+');


def incorporateBatchNorm(self, bn):

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        var_sqrt = torch.sqrt(var + eps)

        w = (self.weight * gamma.reshape(self.out_channels, 1, 1, 1)) / var_sqrt.reshape(self.out_channels, 1,
                                                                                         1, 1)
        b = ((self.bias - mean) * gamma) / var_sqrt + beta

        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)


def get_BatchNorm_parameters(model, file_name):

    file="BN value{}\n";
    write_txt(file_name, file,type='w+');
    
    gamma_arr = []
    beta_arr  = []
    mean_arr  = []
    eps_arr   = []


    for m in model.modules():
        file="m is {}".format(str(m)) ;
        write_txt(file_name, file, type='a+');
        
        if isinstance(m, nn.BatchNorm2d):
            gamma_arr.append( m.weight )
            beta_arr.append (m.bias    )
            mean_arr.append (m.running_mean)
            eps_arr.append  (m.eps)
    
    
    for m in range(0, len(gamma_arr)):
        
        file="m is {}".format(m) ;
        write_txt(file_name, file, type='a+');
        
        file="max gamma_arr[{}] is {}".format(m, max(gamma_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');
        file="min gamma_arr[{}] is {}".format(m, min(gamma_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');
        
        file="max beta_arr[{}] is {}".format(m, max(beta_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');
        file="min beta_arr[{}] is {}".format(m, min(beta_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');
        
        file="max mean_arr[{}] is {}".format(m, max(mean_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');
        file="min mean_arr[{}] is {}".format(m, min(mean_arr[m]).item() ) ;
        write_txt(file_name,  file, type='a+');
        
        file="eps_arr[{}] is {}".format(m, eps_arr[m] ) ;
        write_txt(file_name,  file, type='a+');
        file="eps_arr[{}] is {}".format(m, eps_arr[m] ) ;
        write_txt(file_name,  file, type='a+');
    
    return gamma_arr, beta_arr, mean_arr, eps_arr

def get_Conv_parameters(model, file_name):

    file="Con value{}\n";
    write_txt(file_name, file,type='w+');
    
    w_arr = []
    b_arr  = []
  
    for m in model.modules():
        file="m is {}".format(str(m)) ;
        write_txt(file_name, file, type='a+');
        
        if isinstance(m, nn.Conv2d):
            w_arr.append( m.weight )
            
            if m.bias is not None:
                b_arr.append (m.bias    )
    
    
    for m in range(0, len(w_arr)):
        
        file="m is {}".format(m) ;
        write_txt(file_name, file, type='a+');
        
        file="max w[{}] is {}".format(m, torch.max(w_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');
        file="min w[{}] is {}".format(m, torch.min(w_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');

    for m in range(0, len(b_arr)):        
        file="max b[{}] is {}".format(m, torch.max(b_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');
        file="min b[{}] is {}".format(m, torch.min(b_arr[m]).item() ) ;
        write_txt(file_name, file, type='a+');
    
    return w_arr, b_arr




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
# =============================================================================
# =============================================================================
# # DEM_net_2, wu and 2019
# =============================================================================
# =============================================================================

class  DEM_net_2(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        self.is_deconv     = is_deconv
        self.in_channels   = in_channels
        self.is_batchnorm  = is_batchnorm
        self.n_classes     = ou_channels
        self.filter_num    = filter_num
        
        # filters = [64, 128, 256, 512, 1024]
        # filters = [32, 64, 128, 256, 512]
        # filters = [32, 64, 128, 128, 128]
        filters = [self.filter_num, self.filter_num, self.filter_num, self.filter_num, self.filter_num]
        
        self.up1   = upconv_LeakyReLU_1(self.in_channels, filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=1);
        self.up2   = upconv_LeakyReLU_5(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=1);
        
        
        self.up3   = upconv_LeakyReLU_1(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=2);
        self.up4   = upconv_LeakyReLU_4(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=1);
        
        
        self.up5   = upconv_LeakyReLU_1(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=2);
        self.up6   = upconv_LeakyReLU_4(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=1);
        

        self.up7   = upconv_LeakyReLU_1(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=2);
        self.up8   = upconv_LeakyReLU_4(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=1);
        
        
        self.up9   = upconv_LeakyReLU_1(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=2);
        self.up10   = upconv_LeakyReLU_4(filters[0], filters[0],self.is_batchnorm, kernel_size=kernel_size,stride=1);


        self.final   = upconv(filters[0],self.n_classes, kernel_size=3,stride=2);
        # self.final   = upconv_tanh(filters[0],self.n_classes, kernel_size=3,stride=2);
        
        
    # def forward(self, inputs, label_dsp_dim):
    def forward(self, inputs):
        
        up1 =   self.up1(inputs);   #print(" up1 {}".format( up1.size()));
        up2 =   self.up2(up1);      #print(" up2 {}".format( up2.size()));
        up3 =   self.up3(up2);      #print(" up3 {}".format( up3.size()));
        up4 =   self.up4(up3);      #print(" up4 {}".format( up4.size()));
        up5 =   self.up5(up4);      #print(" up5 {}".format( up5.size()));
        up6 =   self.up6(up5);      #print(" up6 {}".format( up6.size()));
        up7 =   self.up7(up6);      #print(" up7 {}".format( up7.size()));
        up8 =   self.up8(up7);      #print(" up8 {}".format( up8.size()));
        up9 =   self.up9(up8);      #print(" up9 {}".format( up9.size()));
        up10 =  self.up10(up9);     #print(" up10 {}".format(up10.size()));

        final         = self.final(up10);   #print(" up10 {}".format(final.size()));
        
        label_dsp_dim1 = inputs.size(2);
        label_dsp_dim2 = inputs.size(3);
        # label_dsp_dim1 = label_dsp_dim[0]
        # label_dsp_dim2 = label_dsp_dim[1]
        
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        # print("DEM_net_2 final.shape {}".format(final.shape));
        final_clip    = final[:,:,beg1:end1,beg2:end2].contiguous()
        # up1    = up1[:,:,1:1+label_dsp_dim[0],1:1+label_dsp_dim[1]].contiguous()
        
        return final_clip
    
    # Initialization of Parameters
    def _initialize_weights(self):
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



# =============================================================================
# =============================================================================
# # # =============================================================================
# # # DEM_net_3             
# # # =============================================================================
# =============================================================================
# =============================================================================
class DEM_net_3(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
            
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        self.norm1 = nn.BatchNorm2d(filter_num)
        self.norm2 = nn.BatchNorm2d(filter_num)
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.conv1.weight.data.normal_(0, init_weight)
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
         
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
        
    def forward(self, x):
        y1 = self.ac(self.norm1((self.conv1(x))));  
        y2 = self.ac(self.norm2((self.conv2(y1))));
        final = self.conv_final(y2);
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        # print("DEM_net_3 y1.shape {}".format(y1.shape));
        # print("DEM_net_3 y2.shape {}".format(y2.shape));
        # print("DEM_net_3 final.shape {}".format(final.shape));
        # print("DEM_net_3 output.shape {}".format(output.shape));  
        
        return output

    
# =============================================================================
# =============================================================================
# # # =============================================================================
# # # DEM_net_30                    
# # # =============================================================================
# =============================================================================
# =============================================================================
class DEM_net_30(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv3 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv4 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv5 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
        
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv3 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv4 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv5 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.drop_layer = drop_layer
        
        self.drop2 = nn.Dropout(p=drop_out)
        self.drop3 = nn.Dropout(p=drop_out)
        self.drop4 = nn.Dropout(p=drop_out)
        
        self.norm1 = nn.BatchNorm2d(filter_num)
        self.norm2 = nn.BatchNorm2d(filter_num)
        self.norm3 = nn.BatchNorm2d(filter_num)
        self.norm4 = nn.BatchNorm2d(filter_num)
        self.norm5 = nn.BatchNorm2d(filter_num)
        
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        self.conv4.weight.data.normal_(0, init_weight)
        self.conv5.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
        
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
                    
    def forward(self, x):
        
        if self.drop_layer==0:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2))))  ;
            y4 =             self.ac(self.norm4((self.conv4(y3))))  ;
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
                
            # y5 =             self.ac(self.norm5((self.conv5(  self.ac(self.norm4((self.conv4(   self.ac(self.norm3((self.conv3(   self.ac(self.norm2((self.conv2(   self.ac(self.norm1((self.conv1(x))))   ))))   ))))   ))))  ))))  ;
            
        if self.drop_layer==1:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2))))  ;
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            
            # y5 =             self.ac(self.norm5((self.conv5(  self.drop4( self.ac(self.norm4((self.conv4(   self.ac(self.norm3((self.conv3(  self.ac(self.norm2((self.conv2(  self.ac(self.norm1((self.conv1(x))))   ))))   ))))   )))) )  ))))  ;
        
        if self.drop_layer==2:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 = self.drop3( self.ac(self.norm3((self.conv3(y2)))) );
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            
            # y5 =               self.ac(self.norm5((self.conv5(    self.drop4( self.ac(self.norm4((self.conv4(     self.drop3( self.ac(self.norm3((self.conv3(  self.ac(self.norm2((self.conv2(    self.ac(self.norm1((self.conv1(x))))   ))))   )))) )   )))) )   ))))  ;
        
        if self.drop_layer==3:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 = self.drop2( self.ac(self.norm2((self.conv2(y1)))) );
            y3 = self.drop3( self.ac(self.norm3((self.conv3(y2)))) );
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            
            # y5 =             self.ac(self.norm5((self.conv5( self.drop4( self.ac(self.norm4((self.conv4( self.drop3( self.ac(self.norm3((self.conv3(   self.drop2( self.ac(self.norm2((self.conv2(   self.ac(self.norm1((self.conv1(x))))  )))) )  )))) )   )))) )  ))))  ;
            
        final = self.conv_final(y5);
            
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        # print("DEM_net_3 y1.shape {}".format(y1.shape));
        # print("DEM_net_3 y2.shape {}".format(y2.shape));
        # print("DEM_net_3 final.shape {}".format(final.shape));
        # print("DEM_net_3 output.shape {}".format(output.shape));  
        
        return output

# =============================================================================
# =============================================================================
# # # =============================================================================
# # # DEM_net_31                    
# # # =============================================================================
# =============================================================================
# =============================================================================
class DEM_net_31(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv3 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv4 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            
            self.conv5 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
        
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv3 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv4 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv5 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.drop_layer = drop_layer
        
        self.drop2 = nn.Dropout(p=drop_out)
        self.drop3 = nn.Dropout(p=drop_out)
        self.drop4 = nn.Dropout(p=drop_out)
        
        self.norm1 = nn.BatchNorm2d(filter_num)
        self.norm2 = nn.BatchNorm2d(filter_num)
        self.norm3 = nn.BatchNorm2d(filter_num)
        self.norm4 = nn.BatchNorm2d(filter_num)
        self.norm5 = nn.BatchNorm2d(filter_num)
        
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        self.conv4.weight.data.normal_(0, init_weight)
        self.conv5.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
        
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
                    
    def forward(self, x):
        
        if self.drop_layer==0:
            # y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y1 =             self.ac( (self.conv1(x)) )   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2))))  ;
            y4 =             self.ac(self.norm4((self.conv4(y3))))  ;
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            
        if self.drop_layer==1:
            # y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y1 =             self.ac( (self.conv1(x)) )   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2))))  ;
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            
        if self.drop_layer==2:
            # y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y1 =             self.ac( (self.conv1(x)) )   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 = self.drop3( self.ac(self.norm3((self.conv3(y2)))) );
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
        
        if self.drop_layer==3:
            # y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y1 =             self.ac( (self.conv1(x)) )   ;
            y2 = self.drop2( self.ac(self.norm2((self.conv2(y1)))) );
            y3 = self.drop3( self.ac(self.norm3((self.conv3(y2)))) );
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;

        final = self.conv_final(y5);
            
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        return output


class DEM_net_32(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv3 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv4 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            
            self.conv5 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
        
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv3 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv4 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv5 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.drop_layer = drop_layer
        
        self.drop2 = nn.Dropout(p=drop_out)
        self.drop3 = nn.Dropout(p=drop_out)
        self.drop4 = nn.Dropout(p=drop_out)
        
        # self.norm1 = nn.BatchNorm2d(filter_num)
        # self.norm2 = nn.BatchNorm2d(filter_num)
        # self.norm3 = nn.BatchNorm2d(filter_num)
        # self.norm4 = nn.BatchNorm2d(filter_num)
        # self.norm5 = nn.BatchNorm2d(filter_num)
        
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        self.conv4.weight.data.normal_(0, init_weight)
        self.conv5.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
        
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
                    
    def forward(self, x):
        
        if self.drop_layer==0:
            y1 =             self.ac( ((self.conv1(x))))   ;
            y2 =             self.ac( ((self.conv2(y1))))  ;
            y3 =             self.ac( ((self.conv3(y2))))  ;
            y4 =             self.ac( ((self.conv4(y3))))  ;
            y5 =             self.ac( ((self.conv5(y4))))  ;
            
        if self.drop_layer==1:
            y1 =             self.ac( ((self.conv1(x))))   ;
            y2 =             self.ac( ((self.conv2(y1))))  ;
            y3 =             self.ac( ((self.conv3(y2))))  ;
            y4 = self.drop4( self.ac( ((self.conv4(y3)))) );
            y5 =             self.ac( ((self.conv5(y4))))  ;
        
        if self.drop_layer==2:
            y1 =             self.ac( ((self.conv1(x))))   ;
            y2 =             self.ac( ((self.conv2(y1))))  ;
            y3 = self.drop3( self.ac( ((self.conv3(y2)))) );
            y4 = self.drop4( self.ac( ((self.conv4(y3)))) );
            y5 =             self.ac( ((self.conv5(y4))))  ;
        
        if self.drop_layer==3:
            y1 =             self.ac( ((self.conv1(x))))   ;
            y2 = self.drop2( self.ac( ((self.conv2(y1)))) );
            y3 = self.drop3( self.ac( ((self.conv3(y2)))) );
            y4 = self.drop4( self.ac( ((self.conv4(y3)))) );
            y5 =             self.ac( ((self.conv5(y4))))  ;
            
        final = self.conv_final(y5);
            
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        return output



class DEM_net_33(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv3 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv4 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            
            self.conv5 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
        
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv3 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv4 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv5 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.drop_layer = drop_layer
        
        self.drop2 = nn.Dropout(p=drop_out)
        self.drop3 = nn.Dropout(p=drop_out)
        self.drop4 = nn.Dropout(p=drop_out)
        
        self.norm1 = nn.BatchNorm2d(filter_num)
        self.norm2 = nn.BatchNorm2d(filter_num)
        self.norm3 = nn.BatchNorm2d(filter_num)
        self.norm4 = nn.BatchNorm2d(filter_num)
        self.norm5 = nn.BatchNorm2d(filter_num)
        
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        self.conv4.weight.data.normal_(0, init_weight)
        self.conv5.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
        
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
                    
    def forward(self, x):
        
        if self.drop_layer==0:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2 + x))))  ;
            y4 =             self.ac(self.norm4((self.conv4(y3))))  ;
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            
        if self.drop_layer==1:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2 + x))))  ;
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
        
        if self.drop_layer==2:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 = self.drop3( self.ac(self.norm3((self.conv3(y2 + x)))) );
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
        
        if self.drop_layer==3:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 = self.drop2( self.ac(self.norm2((self.conv2(y1)))) );
            y3 = self.drop3( self.ac(self.norm3((self.conv3(y2 + x)))) );
            y4 = self.drop4( self.ac(self.norm4((self.conv4(y3)))) );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            
        final = self.conv_final(y5);
            
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        # print("DEM_net_3 y1.shape {}".format(y1.shape));
        # print("DEM_net_3 y2.shape {}".format(y2.shape));
        # print("DEM_net_3 final.shape {}".format(final.shape));
        # print("DEM_net_3 output.shape {}".format(output.shape));  
        
        return output

# =============================================================================
# =============================================================================
# # # =============================================================================
# # # DEM_net_40                    
# # # =============================================================================
# =============================================================================
# =============================================================================
class DEM_net_40(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv3 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv4 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv5 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv6 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv7 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv8 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
        
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv3 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv4 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv5 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv6 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv7 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv8 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.drop_layer = drop_layer
        
        self.drop4 = nn.Dropout(p=drop_out)
        self.drop7 = nn.Dropout(p=drop_out)
        
        self.norm1 = nn.BatchNorm2d(filter_num)
        self.norm2 = nn.BatchNorm2d(filter_num)
        self.norm3 = nn.BatchNorm2d(filter_num)
        self.norm4 = nn.BatchNorm2d(filter_num)
        self.norm5 = nn.BatchNorm2d(filter_num)
        self.norm6 = nn.BatchNorm2d(filter_num)
        self.norm7 = nn.BatchNorm2d(filter_num)
        self.norm8 = nn.BatchNorm2d(filter_num)
        
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        self.conv4.weight.data.normal_(0, init_weight)
        self.conv5.weight.data.normal_(0, init_weight)
        self.conv6.weight.data.normal_(0, init_weight)
        self.conv7.weight.data.normal_(0, init_weight)
        self.conv8.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
        
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)
            torch.nn.init.zeros_(self.conv6.bias)
            torch.nn.init.zeros_(self.conv7.bias)
            torch.nn.init.zeros_(self.conv8.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
                    
    def forward(self, x):
        
        if self.drop_layer==0:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2))))  ;
            y4 =             self.ac(self.norm4((self.conv4(y3 + y1))));
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            y6 =             self.ac(self.norm6((self.conv6(y5))))  ;
            y7 =             self.ac(self.norm7((self.conv7(y6 + y4))))  ;
            y8 =             self.ac(self.norm8((self.conv8(y7))))  ;
            
        if self.drop_layer==1:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2))))  ;
            y4 =self.drop4(  self.ac(self.norm4((self.conv4(y3 + y1))))  );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            y6 =             self.ac(self.norm6((self.conv6(y5))))  ;
            y7 =             self.ac(self.norm7((self.conv7(y6 + y4))))  ;
            y8 =             self.ac(self.norm8((self.conv8(y7))))  ;
        
        if self.drop_layer==2:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2))))  ;
            y4 =self.drop4(  self.ac(self.norm4((self.conv4(y3 + y1))))  );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            y6 =             self.ac(self.norm6((self.conv6(y5))))  ;
            y7 =self.drop7(  self.ac(self.norm7((self.conv7(y6 + y4))))  );
            y8 =             self.ac(self.norm8((self.conv8(y7))))  ;

        final = self.conv_final(y8);
            
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        # print("DEM_net_3 y1.shape {}".format(y1.shape));
        # print("DEM_net_3 y2.shape {}".format(y2.shape));
        # print("DEM_net_3 final.shape {}".format(final.shape));
        # print("DEM_net_3 output.shape {}".format(output.shape));  
        
        return output


class DEM_net_41(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv3 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv4 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv5 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv6 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv7 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv8 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
        
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv3 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv4 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv5 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv6 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv7 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv8 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.drop_layer = drop_layer
        
        self.drop4 = nn.Dropout(p=drop_out)
        self.drop7 = nn.Dropout(p=drop_out)
        
        self.norm1 = nn.BatchNorm2d(filter_num)
        self.norm2 = nn.BatchNorm2d(filter_num)
        self.norm3 = nn.BatchNorm2d(filter_num)
        self.norm4 = nn.BatchNorm2d(filter_num)
        self.norm5 = nn.BatchNorm2d(filter_num)
        self.norm6 = nn.BatchNorm2d(filter_num)
        self.norm7 = nn.BatchNorm2d(filter_num)
        self.norm8 = nn.BatchNorm2d(filter_num)
        
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        self.conv4.weight.data.normal_(0, init_weight)
        self.conv5.weight.data.normal_(0, init_weight)
        self.conv6.weight.data.normal_(0, init_weight)
        self.conv7.weight.data.normal_(0, init_weight)
        self.conv8.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
        
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)
            torch.nn.init.zeros_(self.conv6.bias)
            torch.nn.init.zeros_(self.conv7.bias)
            torch.nn.init.zeros_(self.conv8.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
                    
    def forward(self, x):
        
        if self.drop_layer==0:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2 + x))))  ;
            y4 =             self.ac(self.norm4((self.conv4(y3))));
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            y6 =             self.ac(self.norm6((self.conv6(y5 + y3 + x))))  ;
            y7 =             self.ac(self.norm7((self.conv7(y6))))  ;
            y8 =             self.ac(self.norm8((self.conv8(y7))))  ;
            
        if self.drop_layer==1:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2 + x))))  ;
            y4 =self.drop4(  self.ac(self.norm4((self.conv4(y3))))  );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            y6 =             self.ac(self.norm6((self.conv6(y5 + y3 + x))))  ;
            y7 =             self.ac(self.norm7((self.conv7(y6))))  ;
            y8 =             self.ac(self.norm8((self.conv8(y7))))  ;
        
        if self.drop_layer==2:
            y1 =             self.ac(self.norm1((self.conv1(x))))   ;
            y2 =             self.ac(self.norm2((self.conv2(y1))))  ;
            y3 =             self.ac(self.norm3((self.conv3(y2 + x))))  ;
            y4 =self.drop4(  self.ac(self.norm4((self.conv4(y3))))  );
            y5 =             self.ac(self.norm5((self.conv5(y4))))  ;
            y6 =             self.ac(self.norm6((self.conv6(y5 + y3 + x))))  ;
            y7 =self.drop7(  self.ac(self.norm7((self.conv7(y6))))  );
            y8 =             self.ac(self.norm8((self.conv8(y7))))  ;

        final = self.conv_final(y8);
            
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        # print("DEM_net_3 y1.shape {}".format(y1.shape));
        # print("DEM_net_3 y2.shape {}".format(y2.shape));
        # print("DEM_net_3 final.shape {}".format(final.shape));
        # print("DEM_net_3 output.shape {}".format(output.shape));  
        
        return output  


class DEM_net_50(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv3 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv4 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv5 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv6 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv7 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv8 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv9 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv10 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv11 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv12 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv13 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
        
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv3 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv4 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv5 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv6 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv7 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv8 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv9 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv10 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv11 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv12 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv13 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.drop_layer = drop_layer
        
        self.drop5 = nn.Dropout(p=drop_out)
        self.drop9 = nn.Dropout(p=drop_out)
        self.drop10 = nn.Dropout(p=drop_out)
        
        self.norm1  = nn.BatchNorm2d(filter_num)
        self.norm2  = nn.BatchNorm2d(filter_num)
        self.norm3  = nn.BatchNorm2d(filter_num)
        self.norm4  = nn.BatchNorm2d(filter_num)
        self.norm5  = nn.BatchNorm2d(filter_num)
        self.norm6  = nn.BatchNorm2d(filter_num)
        self.norm7  = nn.BatchNorm2d(filter_num)
        self.norm8  = nn.BatchNorm2d(filter_num)
        self.norm9  = nn.BatchNorm2d(filter_num)
        self.norm10 = nn.BatchNorm2d(filter_num)
        self.norm11 = nn.BatchNorm2d(filter_num)
        self.norm12 = nn.BatchNorm2d(filter_num)
        self.norm13 = nn.BatchNorm2d(filter_num)
        
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        self.conv4.weight.data.normal_(0, init_weight)
        self.conv5.weight.data.normal_(0, init_weight)
        self.conv6.weight.data.normal_(0, init_weight)
        self.conv7.weight.data.normal_(0, init_weight)
        self.conv8.weight.data.normal_(0, init_weight)
        self.conv9.weight.data.normal_(0, init_weight)
        self.conv10.weight.data.normal_(0, init_weight)
        self.conv11.weight.data.normal_(0, init_weight)
        self.conv12.weight.data.normal_(0, init_weight)
        self.conv13.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
        
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)
            torch.nn.init.zeros_(self.conv6.bias)
            torch.nn.init.zeros_(self.conv7.bias)
            torch.nn.init.zeros_(self.conv8.bias)
            torch.nn.init.zeros_(self.conv9.bias)
            torch.nn.init.zeros_(self.conv10.bias)
            torch.nn.init.zeros_(self.conv11.bias)
            torch.nn.init.zeros_(self.conv12.bias)
            torch.nn.init.zeros_(self.conv13.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
                    
    def forward(self, x):
        
        if self.drop_layer==0:
            y1  =             self.ac(self.norm1 ((self.conv1(x))))   ;
            y2  =             self.ac(self.norm2 ((self.conv2(y1))))  ;
            y3  =             self.ac(self.norm3 ((self.conv3(y2))))  ;
            y4  =             self.ac(self.norm4 ((self.conv4(y3))))  ;
            y5  =             self.ac(self.norm5 ((self.conv5(y4   + y1))))  ;
            y6  =             self.ac(self.norm6 ((self.conv6(y5))))  ;
            y7  =             self.ac(self.norm7 ((self.conv7(y6))))  ;
            y8  =             self.ac(self.norm8 ((self.conv8(y7   + y4))))  ;
            y9  =             self.ac(self.norm9 ((self.conv9(y8))))  ;
            y10 =             self.ac(self.norm10((self.conv10(y9))))  ;
            y11 =             self.ac(self.norm11((self.conv11(y10 +  y7))))  ;
            y12 =             self.ac(self.norm12((self.conv12(y11))))  ;
            y13 =             self.ac(self.norm13((self.conv13(y12))))  ;
            
        if self.drop_layer==1:
            y1  =             self.ac(self.norm1 ((self.conv1(x))))   ;
            y2  =             self.ac(self.norm2 ((self.conv2(y1))))  ;
            y3  =             self.ac(self.norm3 ((self.conv3(y2))))  ;
            y4  =             self.ac(self.norm4 ((self.conv4(y3))))  ;
            y5  =self.drop5(  self.ac(self.norm5 ((self.conv5(y4   + y1))))  );
            y6  =             self.ac(self.norm6 ((self.conv6(y5))))  ;
            y7  =             self.ac(self.norm7 ((self.conv7(y6))))  ;
            y8  =             self.ac(self.norm8 ((self.conv8(y7   + y4))))  ;
            y9  =             self.ac(self.norm9 ((self.conv9(y8))))  ;
            y10 =             self.ac(self.norm10((self.conv10(y9))))  ;
            y11 =             self.ac(self.norm11((self.conv11(y10 +  y7))))  ;
            y12 =             self.ac(self.norm12((self.conv12(y11))))  ;
            y13 =             self.ac(self.norm13((self.conv13(y12))))  ;
        
        if self.drop_layer==2:
            y1  =             self.ac(self.norm1 ((self.conv1(x))))   ;
            y2  =             self.ac(self.norm2 ((self.conv2(y1))))  ;
            y3  =             self.ac(self.norm3 ((self.conv3(y2))))  ;
            y4  =             self.ac(self.norm4 ((self.conv4(y3))))  ;
            y5  =self.drop5(  self.ac(self.norm5 ((self.conv5(y4   + y1))))  );
            y6  =             self.ac(self.norm6 ((self.conv6(y5))))  ;
            y7  =             self.ac(self.norm7 ((self.conv7(y6))))  ;
            y8  =             self.ac(self.norm8 ((self.conv8(y7   + y4))))  ;
            y9  =             self.ac(self.norm9 ((self.conv9(y8))))  ;
            y10 =self.drop10(  self.ac(self.norm10((self.conv10(y9))))  );
            y11 =             self.ac(self.norm11((self.conv11(y10 +  y7))))  ;
            y12 =             self.ac(self.norm12((self.conv12(y11))))  ;
            y13 =             self.ac(self.norm13((self.conv13(y12))))  ;

        final = self.conv_final(y13);
            
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        # print("DEM_net_3 y1.shape {}".format(y1.shape));
        # print("DEM_net_3 y2.shape {}".format(y2.shape));
        # print("DEM_net_3 final.shape {}".format(final.shape));
        # print("DEM_net_3 output.shape {}".format(output.shape));  
        
        return output  

class DEM_net_51(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv2 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv3 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv4 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv5 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv6 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv7 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv8 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv9 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv10 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv11 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            self.conv12 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            self.conv13 = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1) )
            
            # self.conv_final = nn.utils.parametrizations.spectral_norm( nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True) )
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2) 
        
        else:
            self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv3 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv4 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv5 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv6 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv7 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv8 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv9 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv10 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv11 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            self.conv12 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv13 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool1)
            
            self.conv_final = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=bias_bool2)
        
        
        if Leaky==0:
            self.ac = nn.ReLU(inplace=True);
        else:
            self.ac = nn.LeakyReLU(negative_slope = Leaky, inplace=True);
        
        self.drop_layer = drop_layer
        
        self.drop5 = nn.Dropout(p=drop_out)
        self.drop9 = nn.Dropout(p=drop_out)
        self.drop10 = nn.Dropout(p=drop_out)
        
        self.norm1  = nn.BatchNorm2d(filter_num)
        self.norm2  = nn.BatchNorm2d(filter_num)
        self.norm3  = nn.BatchNorm2d(filter_num)
        self.norm4  = nn.BatchNorm2d(filter_num)
        self.norm5  = nn.BatchNorm2d(filter_num)
        self.norm6  = nn.BatchNorm2d(filter_num)
        self.norm7  = nn.BatchNorm2d(filter_num)
        self.norm8  = nn.BatchNorm2d(filter_num)
        self.norm9  = nn.BatchNorm2d(filter_num)
        self.norm10 = nn.BatchNorm2d(filter_num)
        self.norm11 = nn.BatchNorm2d(filter_num)
        self.norm12 = nn.BatchNorm2d(filter_num)
        self.norm13 = nn.BatchNorm2d(filter_num)
        
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        self.conv4.weight.data.normal_(0, init_weight)
        self.conv5.weight.data.normal_(0, init_weight)
        self.conv6.weight.data.normal_(0, init_weight)
        self.conv7.weight.data.normal_(0, init_weight)
        self.conv8.weight.data.normal_(0, init_weight)
        self.conv9.weight.data.normal_(0, init_weight)
        self.conv10.weight.data.normal_(0, init_weight)
        self.conv11.weight.data.normal_(0, init_weight)
        self.conv12.weight.data.normal_(0, init_weight)
        self.conv13.weight.data.normal_(0, init_weight)
        self.conv_final.weight.data.normal_(0, init_weight)
        
        if bias_bool1:
            torch.nn.init.zeros_(self.conv1.bias)
            torch.nn.init.zeros_(self.conv2.bias)
            torch.nn.init.zeros_(self.conv3.bias)
            torch.nn.init.zeros_(self.conv4.bias)
            torch.nn.init.zeros_(self.conv5.bias)
            torch.nn.init.zeros_(self.conv6.bias)
            torch.nn.init.zeros_(self.conv7.bias)
            torch.nn.init.zeros_(self.conv8.bias)
            torch.nn.init.zeros_(self.conv9.bias)
            torch.nn.init.zeros_(self.conv10.bias)
            torch.nn.init.zeros_(self.conv11.bias)
            torch.nn.init.zeros_(self.conv12.bias)
            torch.nn.init.zeros_(self.conv13.bias)
        if bias_bool2:
            torch.nn.init.zeros_(self.conv_final.bias)
                    
    def forward(self, x):
        
        if self.drop_layer==0:
            y1  =             self.ac(self.norm1 ((self.conv1(x))))   ;
            y2  =             self.ac(self.norm2 ((self.conv2(y1))))  ;
            y3  =             self.ac(self.norm3 ((self.conv3(y2   + x))))  ;
            y4  =             self.ac(self.norm4 ((self.conv4(y3))))  ;
            y5  =             self.ac(self.norm5 ((self.conv5(y4))))  ;
            y6  =             self.ac(self.norm6 ((self.conv6(y5   + y3 + x))))  ;
            y7  =             self.ac(self.norm7 ((self.conv7(y6))))  ;
            y8  =             self.ac(self.norm8 ((self.conv8(y7))))  ;
            y9  =             self.ac(self.norm9 ((self.conv9(y8   + y5 + x))))  ;
            y10 =             self.ac(self.norm10((self.conv10(y9))))  ;
            y11 =             self.ac(self.norm11((self.conv11(y10))))  ;
            y12 =             self.ac(self.norm12((self.conv12(y11 + y8 + x))))  ;
            y13 =             self.ac(self.norm13((self.conv13(y12))))  ;
            
        if self.drop_layer==1:
            y1  =             self.ac(self.norm1 ((self.conv1(x))))   ;
            y2  =             self.ac(self.norm2 ((self.conv2(y1))))  ;
            y3  =             self.ac(self.norm3 ((self.conv3(y2   + x))))  ;
            y4  =             self.ac(self.norm4 ((self.conv4(y3))))  ;
            y5  =self.drop5(  self.ac(self.norm5 ((self.conv5(y4))))  );
            y6  =             self.ac(self.norm6 ((self.conv6(y5   + y3 + x))))  ;
            y7  =             self.ac(self.norm7 ((self.conv7(y6))))  ;
            y8  =             self.ac(self.norm8 ((self.conv8(y7))))  ;
            y9  =             self.ac(self.norm9 ((self.conv9(y8   + y5 + x))))  ;
            y10 =             self.ac(self.norm10((self.conv10(y9))))  ;
            y11 =             self.ac(self.norm11((self.conv11(y10))))  ;
            y12 =             self.ac(self.norm12((self.conv12(y11 + y8 + x))))  ;
            y13 =             self.ac(self.norm13((self.conv13(y12))))  ;
        
        if self.drop_layer==2:
            y1  =             self.ac(self.norm1 ((self.conv1(x))))   ;
            y2  =             self.ac(self.norm2 ((self.conv2(y1))))  ;
            y3  =             self.ac(self.norm3 ((self.conv3(y2   + x))))  ;
            y4  =             self.ac(self.norm4 ((self.conv4(y3))))  ;
            y5  =self.drop5(  self.ac(self.norm5 ((self.conv5(y4))))  );
            y6  =             self.ac(self.norm6 ((self.conv6(y5   + y3 + x))))  ;
            y7  =             self.ac(self.norm7 ((self.conv7(y6))))  ;
            y8  =             self.ac(self.norm8 ((self.conv8(y7))))  ;
            y9  =             self.ac(self.norm9 ((self.conv9(y8   + y5 + x))))  ;
            y10 =self.drop10(             self.ac(self.norm10((self.conv10(y9))))  );
            y11 =             self.ac(self.norm11((self.conv11(y10))))  ;
            y12 =             self.ac(self.norm12((self.conv12(y11 + y8 + x))))  ;
            y13 =             self.ac(self.norm13((self.conv13(y12))))  ;

        final = self.conv_final(y13);
            
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        # print("DEM_net_3 y1.shape {}".format(y1.shape));
        # print("DEM_net_3 y2.shape {}".format(y2.shape));
        # print("DEM_net_3 final.shape {}".format(final.shape));
        # print("DEM_net_3 output.shape {}".format(output.shape));  
        
        return output  







# =============================================================================
# =============================================================================
# # # =============================================================================
# # # DEM_net_300                    
# # # =============================================================================
# =============================================================================
# =============================================================================
class DEM_net_300(nn.Module):
    def __init__(self, in_channels=1, ou_channels=1, filter_num=32, kernel_size=(3,3), padding=(3,3), padding_mode='replicate', is_deconv=False, is_batchnorm=False, spectral_norm=False, init_weight=0.0005, bias_bool1=True, bias_bool2=True, Leaky=0.01, neu_in_num=500, neu_ou_num=500, drop_layer=2, drop_out=0.5, neu_ini=0.001):
        super().__init__()
        
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
        
        self.drop_layer = drop_layer
        self.neu_in_num = neu_in_num
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(neu_in_num, neu_ou_num)
        
        self.drop1 = nn.Dropout(p=drop_out)
        self.drop2 = nn.Dropout(p=drop_out)
        self.drop3 = nn.Dropout(p=drop_out)
        self.drop4 = nn.Dropout(p=drop_out)
        
        if self.drop_layer == 2:
            self.fc2 = nn.Linear(neu_ou_num, neu_in_num)
        else:
            self.fc2 = nn.Linear(neu_ou_num, neu_ou_num)
        
        if self.drop_layer == 3:
            self.fc3 = nn.Linear(neu_ou_num, neu_in_num)
        else:
            self.fc3 = nn.Linear(neu_ou_num, neu_ou_num)
            
        if self.drop_layer == 4:
            self.fc4 = nn.Linear(neu_ou_num, neu_in_num)
            
        
        self.conv1 = nn.Conv2d(in_channels, filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=True)
        self.conv2 = nn.Conv2d(filter_num,  filter_num, kernel_size, padding=padding, padding_mode=padding_mode, bias=True)
        self.conv3 = nn.Conv2d(filter_num, ou_channels, kernel_size, padding=padding, padding_mode=padding_mode, bias=True)
         
        self.norm1 = nn.BatchNorm2d(filter_num)
        self.norm2 = nn.BatchNorm2d(filter_num)
        
        self.conv1.weight.data.normal_(0, init_weight)   #best for 64 filters, 0.0005
        self.conv2.weight.data.normal_(0, init_weight)
        self.conv3.weight.data.normal_(0, init_weight)
        
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
        
        
        
        self.fc1.weight.data.normal_(0, neu_ini)
        torch.nn.init.zeros_(self.fc1.bias)
        
        if self.drop_layer >= 2:
            self.fc2.weight.data.normal_(0, neu_ini)
            torch.nn.init.zeros_(self.fc2.bias)
        if self.drop_layer >= 3:
            self.fc3.weight.data.normal_(0, neu_ini)
            torch.nn.init.zeros_(self.fc3.bias)
        if self.drop_layer >= 4:
            self.fc4.weight.data.normal_(0, neu_ini)
            torch.nn.init.zeros_(self.fc4.bias)
                    
    def forward(self, x):
        
        input_ = self.flatten(x)
        # print("input_.shape {}".format(input_.shape));
        # print("self.neu_in_num {}".format(self.neu_in_num))
        
        tmp_in = self.fc1(input_);
        # print("self.fc1(input_).shape {}".format(tmp_in.shape));
        
        if self.drop_layer==2:
            fc_output = self.fc2( self.drop1( tmp_in ) )
            
        if self.drop_layer==3:
            fc_output = self.fc3( self.drop2( self.fc2( self.drop1( tmp_in ) ) ) )    
            
        if self.drop_layer==4:
            fc_output = self.fc4( self.drop3( self.fc3( self.drop2( self.fc2( self.drop1( tmp_in ) ) ) ) ) );
        
        # print("x.shape {}".format(x.shape));
        # print("fc_output.shape {}".format(fc_output.shape));
        fc_4D = fc_output.view(x.shape);
        # print("fc_4D.shape {}".format(fc_4D.shape));
        
        y1 = F.relu(self.norm1((self.conv1(fc_4D))));
        # print("y1.shape {}".format(y1.shape));    
        y2 = F.relu(self.norm2((self.conv2(y1))));
        # print("y2.shape {}".format(y2.shape));
        final = self.conv3(y2);
        
        label_dsp_dim1 = x.size(2);
        label_dsp_dim2 = x.size(3);
        
        size1 = final.size(2);
        size2 = final.size(3);
        
        offset1 = size1-label_dsp_dim1;
        offset2 = size2-label_dsp_dim2;
        
        beg1=offset1//2;   end1=beg1+label_dsp_dim1;
        beg2=offset2//2;   end2=beg2+label_dsp_dim2;
        
        output    = final[:,:,beg1:end1,beg2:end2].contiguous()
        
        # print("DEM_net_3 y1.shape {}".format(y1.shape));
        # print("DEM_net_3 y2.shape {}".format(y2.shape));
        # print("DEM_net_3 final.shape {}".format(final.shape));
        # print("DEM_net_3 output.shape {}".format(output.shape));  
        
        return output