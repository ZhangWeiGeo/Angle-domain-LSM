#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 14:33:18 2023

@author: zhangjiwei
"""

import os
from datetime import datetime
import time
import struct
import mmap
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.backends.backend_agg as agg
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, MaxNLocator
import math
#import cv2
import numpy as np
import threading
from PIL import Image
from wiggle.wiggle import wiggle


def past_image_and_colorbar(file1, file2, file3, ratio):
    
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    
    w1, h1 = image1.size
    w2, h2 = image2.size
    
    length    = np.int(w1*ratio)
    total_lenth = w1 + length
    
    print("w1:{},h1:{}".format(w1, h1));print("w2:{},h2:{}".format(w2, h2))
    
    combine = Image.new('RGB', (total_lenth, max(h1,h2)), 'white')

    combine.paste( image2, (length,0) )

    # combine.paste( image1, (0,0) )
    
    combine.save(file3)
    

def mkdir(file_name):
    if not os.path.isdir(file_name):
        #os.makedirs(file_name);
        os.system("mkdir {}".format(file_name));

def rename(file_name,file_name1):
    if os.path.exists(file_name):
        #os.rename(file_name,file_name1);
        os.system("mv {}  {}".format(file_name,file_name1));
        
def cp_r(file_name,file_name1):  
    zzz = "cp -r {}  {}".format(file_name,file_name1);
    os.system(zzz);        

def rm_rf(file_name):
    ###
    zzz = "rm -rf {} ".format(file_name);
    os.system(zzz);
    
def write_txt(file_name,output_some,type='w+'):
    data=open(file_name,type); ##'w+' 'a+'
    print(output_some,file=data)
    data.close()

def write_np_txt(file_name, arr, delimiter='\n', fmt='%.5f'):
    np.savetxt(file_name, np.asarray(arr), delimiter=delimiter, fmt=fmt);

def return_bin_name(input_arr):
    shape = list(input_arr.shape)
    file=""
    for i, j in enumerate(shape): 
        if i < len(shape)-1:
            file = file + str(j) + "-"
        else:
            file = file + str(j)
    return file

def fwrite_file(file_name, x_data, time_bool=True):
    t_start = time.perf_counter();
    
    with open(file_name, 'wb') as f:
        x_data.astype(np.float32).tofile(f)###it is note that c langauge is row first
    
    t_end = time.perf_counter();
    if time_bool:
        print(file_name + " " + str((t_end - t_start))    + " s (write)"    )

def fread_file(file_name, x_data=0, shape_list=0, way=1, offset=0, time_bool=True):
    """
    file_name: data name
    x_data: x_data=0
    shape_list: if shape_list==0, directly retrun array, in_arr = P.fread_file(file_list[i]); Otherwise, example: in_arr = np.zeros((nx,ny,nz),dtype=np.float32);    P.fread_file(in_file, in_arr, (nx,ny,nz));
    way: defalut 1, different way to read the data.
    offest:    uesless
    time_bool: True. output the time of read data
    #noted we must use x_data[:] rather than x_data =  1.0*np.array,  the first operation is set value, the second is to create a new array and the work domain is only the function.
    
    Example: in_arr = P.fread_file(file_list[i]);
    Example: in_arr      = np.zeros((nx,ny,nz),dtype=np.float32);    P.fread_file(in_file, in_arr, (nx,ny,nz));
    """
    
    t_start = time.perf_counter();
    
    if way==0:
        with open(file_name, 'rb') as f:
            arr_1d = np.fromfile(f, dtype=np.float32) 
            # x_data[:] = 1.0*np.array(arr_1d, dtype=(np.float32)).reshape(shape_list) ;###it is note that c langauge is row first
        
    elif way==1:
        with open(file_name, 'rb') as f:
            arr_1d = np.frombuffer(f.read(), dtype=np.float32)
            # x_data[:] = 1.0*np.array(arr_1d, dtype=(np.float32)).reshape(shape_list) ;###it is note that c langauge is row first
        
        
    elif way==2:
        with open(file_name, 'r+b') as f:
            # Memory-map the file using mmap.mmap()
            mmapped = mmap.mmap(f.fileno(), 0)
            # Create a 1D NumPy array from the memory-mapped buffer, using dtype=np.float32
            arr_1d = np.frombuffer(mmapped, dtype=np.float32)
            # x_data[:] = 1.0*np.array(arr_1d, dtype=(np.float32)).reshape(shape_list) ;###it is note that c langauge is row first
    else:
        length=1
        for i in range(0, len(shape_list) ):
            length = length * shape_list[i];
            
        with open(file_name, 'rb') as f:
            arr_1d = struct.unpack( 'f' * (length), f.read())
            # x_data[:] = 1.0*np.array(arr_1d, dtype=(np.float32)).reshape(shape_list) ;###it is note that c langauge is row first
    
    t_end = time.perf_counter();
    
    if time_bool:
        print(file_name + " " +  str((t_end - t_start))   + " s (read)"  )
    
    if shape_list!=0:
        x_data[:] = arr_1d.reshape(shape_list) ;
    else:
        return arr_1d[:]
    

def fread_file_list(file_name, x_data, shape_list, way=0, thread_num=24, time=False):
    
    nz_number = len(file_name);
    
    if (nz_number%thread_num) ==0:
        number = nz_number//thread_num
    else:
        number = nz_number//thread_num + 1
    
    for i in range(0, number):
        file_list = file_name[i*thread_num:(i+1)*thread_num]
        threads_list = []
        
        for t in range(0, thread_num):
            iz = i*thread_num + t
            
            if iz < len(file_name):
                file = file_list[t]
                thr  = threading.Thread(target=(fread_file), args=(file, x_data[:,:,:,iz], shape_list, 0) ); #fread_file(file_name, x_data, shape_list, way=0)
                
                threads_list.append(thr)
                
        for thr in threads_list:
            thr.start()
        for thr in threads_list:
            thr.join();


def plot_graph(input_array, plot_number=1, dz=1, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1="x", label2="y", figsize=(10, 5), axis_width=1, axis_length=1, linewidth=1, linestyle=("-", "-", "-"), grid_linewidth=0, line_color=("k", "r", "b", ), FontSize=(9,9), fontsize=(9,9), legend_size=(9,9), legend_position="best", legend_name=("1","2","3", ), eps_dpi=300, output_name="tmp.eps", title="", plot_end=0, plot2_end=0, reverse_1=False, reverse_2=False, pltshow=False, xscale=False, yscale=False, label1_scientific=0, y_ticks_arr=(0,), powerlimits=(-1,1), y_ticks_arr_num=5):
    """
    input_array: input_arry, list of numpy array, list[0]=np.array
    plot_number: how many number of array is plotted in a figure. We can plot different length (but the same original position and the different end position)
    dz:
    x1beg:
    d1num:
    x2beg:
    d2num:
    
    legend_position: supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    
    pltshow: two options: True: plotshow and save figure, False: only save
    example1: P.plot_graph(in_arr, plot_number=plot_number, dz=dz, x1beg=f1, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1=label1, label2=label2, figsize=figsize1, axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=10, FontSize=10, legend_size=12, legend_position=legend_position, legend_name=legend_name, eps_dpi=300, output_name=ou_file);
    
    P.plot_graph(in_arr, plot_number=plot_number, dz=dz, x1beg=f1, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1=label1, label2=label2, figsize=figsize1, axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=10, FontSize=10, legend_size=12, legend_position=legend_position, legend_name=legend_name, eps_dpi=300, output_name=ou_file, plot_end=plot_end);
    """
    
    x_list=[]
    for i in range(0, plot_number):
        shape = input_array[i].shape
        nz    = shape[0]

        if     x1beg != 0 and x1end==0:
            x_array=np.linspace(x1beg, x1beg+nz*dz, nz)
        elif   x1beg != 0 and x1end!=0:
            x_array=np.linspace(x1beg, x1end, nz)
        elif   x1beg == 0 and x1end!=0:
            x_array=np.linspace(x1beg, x1end, nz)
        else:
            x_array=np.linspace(0, nz*dz, nz)
    
        x_list.append(x_array);
    
	# print_numpy_array_info(x_array, "x_array of plot_graph")
    print("max of x_array is", np.max(x_array) ); print("min of x_array is", np.min(x_array) );
    
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=(10,5))
    
    # Remove the margins of the subplot
    ax.margins(0)
    # Remove all the whitespace around the figure
    fig.set_constrained_layout(True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    for i in range(0, len(x_list)):
        if plot_number>i:
            plt.plot(x_list[i], input_array[i],  line_color[i]+linestyle[i] , linewidth=linewidth)


    plt.xlabel(label1, fontsize=FontSize[0], color="k")
    plt.ylabel(label2, fontsize=FontSize[1], color="k")
    
    
    if d1num!=0:
        plt.xlim(x1beg, x1end)
        num1 = int((x1end - x1beg) / d1num )
        xtick = np.linspace(x1beg,  x1beg + num1*d1num, num1+1)
        plt.xticks(xtick, fontsize=fontsize[0])
        
    if x2beg!=0 and x2end!=0 and d2num!=0:
        plt.ylim(x2beg, x2end)
        num2 = int((x2end - x2beg) / d2num ) 
        ytick = np.linspace(x2beg, x2beg + num2*d2num, num2+1)
        plt.yticks(ytick, fontsize=fontsize[1])
    
    if plot_end!=0:
        plt.xlim(x1beg, plot_end)
    if plot2_end!=0:
        plt.ylim(x2beg, plot2_end)
    
    if xscale:
        plt.xscale('log')  # 设置横坐标为对数比例
    if yscale:
        plt.yscale('log')  # 设置纵坐标为对数比例
    
    
    # get current axis
    ax = plt.gca()
    # get current axis
    ax = plt.gca()
    xtick = ax.get_xticks()
    plt.xticks(fontsize=fontsize[0])
    ytick = ax.get_yticks()
    plt.yticks(fontsize=fontsize[1]) 
    
    # x y
    if axis_length!=0 and axis_width!=0:
        ax.tick_params(axis="both", direction="out", length=axis_length, width=axis_width, colors='k')
        ax.xaxis.set_ticks_position("bottom") #"top","bottom",
        ax.yaxis.set_ticks_position("left")  #"left","right"
        # ax.spines['right'].set_position(("data", 0.01))  #y轴在x轴的0.01

    if label1_scientific:
        
        # if y_ticks_arr_num >=0:
        #     y_ticks_arr = np.arange(x2beg, x2end, d2num);    
        # # if len(y_ticks_arr)<=1:
        # #     ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks_arr_num)) ## 5
        # else:
        #     ticks=ax.yaxis.set_ticks(y_ticks_arr)
        
        print("initial  yaxis  is", ax.yaxis.get_ticklocs())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        
        tick_format.set_powerlimits( powerlimits )
        
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1f'
        ax.yaxis.set_major_formatter(tick_format)
       
        print("final  yaxis  is", ax.yaxis.get_ticklocs())
        

    #  #"top","bottom","left","right"
    if axis_width==0:
        ax.spines["top"].set_linewidth(axis_width)
        ax.spines["bottom"].set_linewidth(axis_width)
        ax.spines["left"].set_linewidth(axis_width)
        ax.spines["right"].set_linewidth(axis_width)
    
    if reverse_1:
        # Reverse the x-axis
        ax.invert_xaxis()
    if reverse_2:
        # Reverse the x-axis
        ax.invert_yaxis()    
    
    if title:
        plt.title(title);
        
    # grid parameters
    if grid_linewidth!=0:
        plt.grid(True, which='major', axis='both', color='k', linestyle='-',  linewidth=grid_linewidth)
        ax.xaxis.grid(color='k', linestyle='--', linewidth=grid_linewidth)
        ax.yaxis.grid(color='k', linestyle='--', linewidth=grid_linewidth)

    if legend_size[0] != 0:
        plt.legend(legend_name, loc=legend_position, fontsize=legend_size[0])
    
    plt.savefig(output_name + ".eps", dpi=eps_dpi)
    plt.savefig(output_name + ".jpg", dpi=eps_dpi)

    
    if not pltshow:
        plt.close();
        
        

def plot_graph_multi(input_array, plot_number=1, dz=1, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=1, x2beg_new=0, d2num_new=1, d2num_new_plot=2, d2label="Angle", label1="", label2="y", figsize=(10, 5), axis_width=1, axis_length=1, linewidth=1, linestyle=("-", "-", "-"), grid_linewidth=0, line_color=("k", "r", "b", ), FontSize=(9,9), fontsize=(9,9), legend_size=(9,9), legend_position="best", legend_name=("1","2","3", ), eps_dpi=300, output_name="tmp.eps", title="", plot_end=0, plot2_end=0, reverse_1=False, reverse_2=False, pltshow=False, xscale=False, yscale=False, label1_scientific=0, y_ticks_arr=(0,), powerlimits=(-1,1), y_ticks_arr_num=5):

### ax.tick_params(axis='x', which='both', direction='out', labelbottom=True, labeltop=True, bottom=True, top=True, width=axis_length, colors='k', labelsize=fontsize[0])
###  which='both'：表示要设置主刻度和次刻度。 direction='out'：指定刻度线的方向为朝外，即刻度线向着数据区域之外延伸。labelbottom=True, labeltop=True：表示同时显示刻度标签在底部和顶部，即在 x 轴的下方和上方都显示刻度标签。 # 显示上方和下方的刻度线# ax.xaxis.set_ticks_position('both') ####等于 bottom=True, top=True
#### 需要明白5个东西  以x为例：
# 1 是否需要保留 ticks的刻度（bottom和top,which='both'：表示要设置主刻度和次刻度），
# 2 是否需要保留ticks对应的label（labelbottom和labeltop），
# 3 ticks对应的线宽 width=axis_length，4 颜色colors='k', 5 ticks对应的label的字号：labelsize=fontsize[0]; 
# 6 隐藏边框 ax.yaxis.set_ticks_position('left')  
# 7 单独设置某一个方向的tick ax.yaxis.set_ticks_position('left'), 如果设置为左，右边默认消失，因为存在both选项    
    
    shape = input_array.shape
    nz    = shape[0]

    if     x1beg != 0 and x1end==0:
        x_array=np.linspace(x1beg, x1beg+nz*dz, nz)
    elif   x1beg != 0 and x1end!=0:
        x_array=np.linspace(x1beg, x1end, nz)
    elif   x1beg == 0 and x1end!=0:
        x_array=np.linspace(x1beg, x1end, nz)
    else:
        x_array=np.linspace(0, nz*dz, nz)
    
    if  x2beg!=0 and x2end!=0 and d2num!=0:
        # z_array=np.arange(x2beg, x2end, d2num);
        num2 = np.int((x2end- x2beg)//d2num +1)
        z_array=np.linspace(x2beg, x2end, num2 )
        
    #print(x_array); 
    print(x_array.shape); 
    print("z_array is", z_array)
    
    if plot_number>1:
        fig, axs = plt.subplots(1, plot_number, figsize=figsize)
    else:
        plt.plot(figsize=figsize)
        fig = plt.gcf();  axs = plt.gca();
        fig = [fig,];     axs = [axs,];
    
    shape= input_array.shape;
    
    # 循环画图
    for i in range(min(plot_number, shape[1])):
        
        if i <len(line_color):
            linecolor = line_color[i];
        else:
            linecolor = line_color[0];
        if i <len(linestyle):
            linestyle_t = linestyle[i];
        else:
            linestyle_t = linestyle[0];

        ax = axs[i]
        ax.margins(0)
        ax.plot(input_array[:,i], x_array, color=linecolor, linewidth=linewidth, linestyle=linestyle_t)
        
        if reverse_1:
            # Reverse the x-axis
            ax.invert_xaxis()
        if reverse_2:
            # Reverse the x-axis
            ax.invert_yaxis()  


        if i==0:
            #######xx
            ax.tick_params(axis='x', which='both', direction='out', labelbottom=True, labeltop=True, bottom=True, top=True, width=axis_length, colors='k', labelsize=fontsize[0])
            
            if len(label1)==0:
                ax.xaxis.set_ticks_position('top') #不保留小标签

            ax.set_xlim(x2beg, x2end)
            ax.set_xticks(z_array)
            
#######yyy          
            ax.tick_params(axis='y', labelleft=True, labelright=False, left=True, right=False, width=axis_length, colors='k', labelsize=fontsize[1])
            ax.spines['right'].set_visible(False) 

            # 设置所有轴线的宽度
            for spine in ax.spines.values():
                spine.set_linewidth(axis_width)
            
            ax.set_ylabel(label2, fontsize=FontSize[1], color="k")


        else:
            ##xx
            ax.tick_params(axis='x', which='both', direction='out', labelbottom=False, labeltop=True, bottom=False, top=True, width=axis_length, colors='k', labelsize=fontsize[0])
            # ax.xaxis.set_ticks_position("bottom") #"top","bottom",
            ax.set_xlim(x2beg, x2end)
            ax.set_xticks(z_array)
            
            ##yyy
            ax.tick_params(axis='y', labelleft=False, labelright=False, left=False, right=False, width=axis_length, colors='k', labelsize=fontsize[1])

            ##
            ax.spines['left'].set_visible(False)
            if i==plot_number-1:
                ax.spines['right'].set_visible(True)
            else:
                ax.spines['right'].set_visible(False)
            
            # 设置所有轴线的宽度
            for spine in ax.spines.values():
                spine.set_linewidth(axis_width)
            
        # 自定义 x 轴刻度标签 
        # 设置 x 轴刻度
        if i==0 and len(label1)!=0:
            
            ax.set_xlabel(label1, fontsize=FontSize[1], color="k")
            ##设置  bottom x axis
            ax.set_xticks(z_array)
            ax.set_xticklabels(z_array) 
            
            # # 创建新的轴对象用于顶部刻度标签
            ax_top = ax.twiny()
            
            ax_top.tick_params(axis='x', which='both', direction='out', labelbottom=False, labeltop=True, bottom=False, top=True, width=axis_width, colors='k', labelsize=fontsize[0])
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            
            ax_top.set_xlim(x2beg, x2end)
            ax_top.set_xticks(z_array)
            
            # # 设置顶部刻度位置和标签
            ax_top.set_xticks([0])
            # # 获取当前 ax_top x 轴刻度标签
            xtick_labels = [str(tick) for tick in ax_top.get_xticks()]
            # # 将特定位置的刻度标签设置为你想要的值
            xtick_labels[0] = str(x2beg_new)
            # # 设置 ax_top x 轴刻度标签
            ax_top.set_xticklabels(xtick_labels)
        
        elif i % d2num_new_plot ==0:
            # ax.spines['top'].set_position((str(i*d2num_new), 0))
            ax.set_xticks([0])  # 只保留0位置的刻度
            # 获取当前 x 轴刻度标签
            xtick_labels = [str(tick) for tick in ax.get_xticks()]
            # 将特定位置的刻度标签设置为你想要的值
            xtick_labels[0] = str(x2beg_new + i*d2num_new)

            # 设置 x 轴刻度标签
            ax.set_xticklabels(xtick_labels)

        else:
            ax.tick_params(axis='x', labeltop=False, labelbottom=False, top=False, bottom=False)
        

######添加d2label
        middle  = plot_number // 2
        
        bool1 = bool(middle == i)
        bool2 = bool(i % d2num_new_plot == 0)
        
        if bool1:
            if bool2:
                ax.set_xlabel(d2label, fontsize=FontSize[1], color="k")
                ax.xaxis.set_label_position('top')
            else:
                ax.tick_params(axis='x', labeltop=True, labelbottom=False, top=False, bottom=False)
                ax.set_xticks([0])
            
                ax.set_xlabel(d2label, fontsize=FontSize[1], color="k")
                ax.xaxis.set_label_position('top')

                ax.set_xticklabels(["  "])###是的，您可以使用 ax.set_xticklabels([""]) 来在顶部标签的位置保留一个空间，使得顶部标签不会移动到另一个位置。


    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.0)

    plt.savefig(output_name + ".eps", dpi=eps_dpi)
    plt.savefig(output_name + ".jpg", dpi=eps_dpi)
    # plt.savefig(output_name + ".png", dpi=eps_dpi)


    if not pltshow:
        plt.close();
        

def plot_graph2(input_array, plot_number=1, dz=1, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1="x", label2="y", figsize=(10, 5), axis_width=1, axis_length=1, linewidth=1, linestyle=("-", "-", "-"), grid_linewidth=0, line_color=("k", "r", "b", ), FontSize=(9,9), fontsize=(9,9), legend_size=(9,9), legend_position="best", legend_name=("1","2","3", ), eps_dpi=300, output_name="tmp.eps", title="", plot_end=0, reverse_1=False, reverse_2=False, pltshow=False, grid=False):
    """
    input_array: input_arry, list of numpy array, list[0]=np.array
    plot_number: how many number of array is plotted in a figure. We can plot different length (but the same original position and the different end position)
    dz:
    x1beg:
    d1num:
    x2beg:
    d2num:
    
    legend_position: supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    
    pltshow: two options: True: plotshow and save figure, False: only save
    example1: P.plot_graph(in_arr, plot_number=plot_number, dz=dz, x1beg=f1, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1=label1, label2=label2, figsize=figsize1, axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=10, FontSize=10, legend_size=12, legend_position=legend_position, legend_name=legend_name, eps_dpi=300, output_name=ou_file);
    
    P.plot_graph(in_arr, plot_number=plot_number, dz=dz, x1beg=f1, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, label1=label1, label2=label2, figsize=figsize1, axis_width=1, axis_length=1, linewidth=1, linestyle=linestyle, grid_linewidth=0, line_color=line_color, fontsize=10, FontSize=10, legend_size=12, legend_position=legend_position, legend_name=legend_name, eps_dpi=300, output_name=ou_file, plot_end=plot_end);
    """
    
    x_list=[]
    for i in range(0, plot_number):
        shape = input_array[i].shape
        nz    = shape[0]

        if     x1beg != 0 and x1end==0:
            x_array=np.linspace(x1beg, x1beg+nz*dz, nz)
        elif   x1beg != 0 and x1end!=0:
            x_array=np.linspace(x1beg, x1end, nz)
        elif   x1beg == 0 and x1end!=0:
            x_array=np.linspace(x1beg, x1end, nz)
        else:
            x_array=np.linspace(0, nz*dz, nz)
    
        x_list.append(x_array);
    
	# print_numpy_array_info(x_array, "x_array of plot_graph")
    print("max of x_array is", np.max(x_array) ); print("min of x_array is", np.min(x_array) );
    
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots(figsize=(10,5))
    
    # Remove the margins of the subplot
    ax.margins(0)
    # Remove all the whitespace around the figure
    fig.set_constrained_layout(True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    if plot_number >= 1:
        plt.semilogy(x_list[0], input_array[0],  line_color[0]+linestyle[0] , linewidth=linewidth)
    if plot_number >= 2:
        plt.semilogy(x_list[1], input_array[1],  line_color[1]+linestyle[1] , linewidth=linewidth)
    if plot_number >= 3:
        plt.semilogy(x_list[2], input_array[2],  line_color[2]+linestyle[2] , linewidth=linewidth)
    if plot_number >= 4:
        plt.semilogy(x_list[3], input_array[3],  line_color[3]+linestyle[3] , linewidth=linewidth)
    if plot_number >= 5:
        plt.semilogy(x_list[4], input_array[4],  line_color[4]+linestyle[4] , linewidth=linewidth)


    plt.xlabel(label1, fontsize=FontSize[0], color="k")
    plt.ylabel(label2, fontsize=FontSize[1], color="k")
    
    
    if d1num!=0:
        plt.xlim(x1beg, x1end)
        num1 = int((x1end - x1beg) / d1num )
        xtick = np.linspace(x1beg,  x1beg + num1*d1num, num1+1)
        plt.xticks(xtick, fontsize=fontsize[0])
        
    if x2beg!=0 and x2end!=0 and d2num!=0:
        plt.ylim(x2beg, x2end)
        num2 = int((x2end - x2beg) / d2num ) 
        ytick = np.linspace(x2beg, x2beg + num2*d2num, num2+1)
        plt.yticks(ytick, fontsize=fontsize[1])
    
    if plot_end!=0:
        plt.xlim(x1beg, plot_end)
    
    # get current axis
    ax = plt.gca()
    # get current axis
    ax = plt.gca()
    xtick = ax.get_xticks()
    plt.xticks(fontsize=fontsize[0])
    ytick = ax.get_yticks()
    plt.yticks(fontsize=fontsize[1]) 
    
    # x y
    if axis_length!=0 and axis_width!=0:
        ax.tick_params(axis="both", direction="out", length=axis_length, width=axis_width, colors='k')
        ax.xaxis.set_ticks_position("bottom") #"top","bottom",
        ax.yaxis.set_ticks_position("left")  #"left","right"
        # ax.spines['right'].set_position(("data", 0.01))  #y轴在x轴的0.01

    #  #"top","bottom","left","right"
    if axis_width==0:
        ax.spines["top"].set_linewidth(axis_width)
        ax.spines["bottom"].set_linewidth(axis_width)
        ax.spines["left"].set_linewidth(axis_width)
        ax.spines["right"].set_linewidth(axis_width)
    
    if reverse_1:
        # Reverse the x-axis
        ax.invert_xaxis()
    if reverse_2:
        # Reverse the x-axis
        ax.invert_yaxis()    
	
    if title:
        plt.title(title);
	
    if grid:
        plt.grid(True);
        
    # grid parameters
    if grid_linewidth!=0:
        plt.grid(True, which='major', axis='both', color='k', linestyle='-',  linewidth=grid_linewidth)
        ax.xaxis.grid(color='k', linestyle='--', linewidth=grid_linewidth)
        ax.yaxis.grid(color='k', linestyle='--', linewidth=grid_linewidth)

    if legend_size != 0:
        plt.legend(legend_name, loc=legend_position, fontsize=legend_size)
    
    plt.savefig(output_name + ".eps", dpi=eps_dpi)
    plt.savefig(output_name + ".jpg", dpi=eps_dpi, quality=100)
    # plt.savefig(output_name + ".png", dpi=eps_dpi, quality=100)
    
    if not pltshow:
        plt.close();        


def imshow1(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight', gca_remove=False, colorbar_text=('left', 'bottom'), colorbar_ticks="", colorbar_ticks_num=5, powerlimits=(-1,1),  ):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=(10,5))


    im1   = ax1.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');

    
    if d1num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
    if d2num!=0:
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    if colorbar:  
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) ##label size 
        
        print("len(colorbar_ticks) is {}", len(colorbar_ticks) )
        if len(colorbar_ticks)==0:
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=colorbar_ticks_num)) ## 5
        if len(colorbar_ticks)!=0:
            ticks=cbar.set_ticks(colorbar_ticks)
        
        cbar.ax.tick_params(labelsize=legend_size[1]) ##ticks size
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        # if np.fabs(vmin)>=0.09999999 and np.fabs(vmax)<9.9999999 :
        #     tick_format.set_powerlimits( (-1,1) )
        # else:
        tick_format.set_powerlimits( powerlimits )
        
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1f'
        cbar.ax.yaxis.set_major_formatter(tick_format)
       
        print("final cbar1.get_ticks() is", cbar.get_ticks())
        ticks=cbar.get_ticks()
        
        
        # cbar.set_ticks(ticks)
        # ticks=cbar.get_ticks()
        # for i in range(0, len(ticks) ):
        #     # if len( str(abs(ticks)[0]) ) <2:
        #     if ticks[i] <:
        #         tick_format = ticker.ScalarFormatter(useMathText=True)
        #         tick_format.set_powerlimits( -2, 2 )
        #         tick_format.set_scientific(True)
        #         tick_format.set_useOffset(False)
        #         tick_format.format = '%.1f'
        #         cbar.ax.yaxis.set_major_formatter(tick_format)
                
        
        print("final cbar2.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        
        
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_horizontalalignment(colorbar_text[0]) #'center', 'right', 'left'
        offset_text.set_verticalalignment(colorbar_text[1]) #supported values are 'top', 'bottom', 'center', 'baseline', 'center_baseline'

        print("offset_text.set_position 1 is", offset_text.set_position)
        
    
    if gca_remove:
        plt.gca().remove()
    
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".jpg", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();
        
        
def imshow2(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight', gca_remove=False, colorbar_text=('left', 'bottom') ):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=(10,5))


    im1   = ax1.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');

    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    if colorbar:  
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        
        offset_text.set
        
        offset_text.set_horizontalalignment(colorbar_text[0]) #'center', 'right', 'left'
        offset_text.set_verticalalignment(colorbar_text[1]) #supported values are 'top', 'bottom', 'center', 'baseline', 'center_baseline'

        print("offset_text.set_position 1 is", offset_text.set_position)
        
        # (x0, y0) = offset_text.get_position
        # offset_text.set_position( ( offset_text.x0 +0.2,  offset_text.y0 +0.2) )
        # print("offset_text.set_position 2 is", offset_text.set_position)
        
        # offset_text.set_position( ( (cbar_box.x0+cbar_box.x1)/2, cbar_box.ymax + 0.5) )
        # print("offset_text.set_position 2 is", offset_text.set_position)
        
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    if gca_remove:
        plt.gca().remove()
    # plt.tight_layout()
    # fig.set_constrained_layout(True)
    
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".jpg", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();
  

def imshow222(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight'):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10,5))

    

    ax1   = plt.subplot(121)
    # ax1 =fig.add_axes([0.1, 0.1, 0.4, 0.8])
    
    im1   = ax1.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');

    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    if colorbar:  
        # cbar_ax = plt.subplot(122)
        cbar_ax =fig.add_axes([0.6, 0.1, 0.6, 0.8])
        cbar = plt.colorbar(im1, cax=cbar_ax, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    # plt.gca().remove()
    # plt.tight_layout()
    # fig.set_constrained_layout(True)
    # plt.tight_layout()
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".jpg", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();

def imshow44(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight', GridSpec_ratio=(5, 2)):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10,5))

    
    # gs = GridSpec(1,2, figure=fig)
    

    
    gs = GridSpec(1,1, figure=fig)
    ax_main    = plt.subplot(gs[0])
    # ax_cbar    = plt.subplot(gs[1])
    
    
    # Plot the 2D array in the left subplot
    ax1   = ax_main.axes

    
    im1   = ax_main.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');
    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
    
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)    
    
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis(axis_mark);
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    fig.subplots_adjust(left=-0.05, right=0.95, bottom=-0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    if colorbar:  
        # cbar = plt.colorbar(im1, cax=ax_cbar, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar = plt.colorbar(im1, label=colorbar_label, fraction=0.04, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        canvas = fig.canvas
        # create Agg
        width, height = canvas.get_width_height()
        dpi = canvas.get_renderer().dpi
        renderer = agg.RendererAgg( width, height, dpi )
        
        # get tex box
        offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        offset_text_width  = offset_text_bb.width
        offset_text_height = offset_text_bb.height
        
        x = cbar_box.x0 + cbar_box.width + offset_text_width
        y = cbar_box.y1 + offset_text_bb.height
        offset_text.set_position((x, y))
        print("offset_text_height is", offset_text_height)
        print("offset_text_width is", offset_text_width)
        print("offset_text.set_position is", offset_text.set_position)
    
    # plt.gca().remove()
    # plt.tight_layout()
    # fig.set_constrained_layout(True)
    
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".jpg", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();    
  
def imshow3(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(12,12,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables="", axis_mark='tight', GridSpec_ratio=(9, 1) ):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    
    
    fig = plt.figure(figsize=(10,5))
    gs = GridSpec(1, 2, width_ratios=[GridSpec_ratio[0], GridSpec_ratio[1]])
    
    ax_main    = plt.subplot(gs[0])
    ax_cbar    = plt.subplot(gs[1])
    
    # Plot the 2D array in the left subplot
    
    im1   = ax_main.imshow(input_array);
    ax1   = ax_main.axes
    
        
    if colorbar:  
        cbar = plt.colorbar(im1, cax=ax_cbar, label=colorbar_label, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".jpg", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();
        
# def imshow2(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables=""):
def imshow2_old2(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(6,3.5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False, FontSize=(14,14), fontsize=(14,14), legend_size=(14,14,10), xtick_positions="", xtick_lables="", ytick_positions="", ytick_lables=""):
    """
    input_array:input_arry, numpy array
   
    pltshow: two options: True: plotshow and save figure, False: only save
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    P.imshow1(in_arr, x1beg=fx, x1end=fx+nx*dz, d1num=0, x2beg=fz+nz*dz, x2end=fz, d2num=0, xlabel=label1, ylabel=label2, vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0:
        x1end = input_array.shape[1]
        
    if  x2beg ==0 and x2end ==0:
        x2beg = input_array.shape[0]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10,5))

    
    # Plot the 2D array in the left subplot
    ax1   = fig.add_subplot( )
    
    im1   = plt.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');
    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
        
    if len(xtick_positions)!=0 and len(xtick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.xticks(xtick_positions, xtick_lables)
        print("xtick_positions is ", xtick_positions)
        print("xtick_lables is ", xtick_lables)
    
    if len(ytick_positions)!=0 and len(ytick_lables)!=0:
        # ax1.set_xticks(xtick_positions);
        # ax1.xaxis.set_major_locator(ticker.FixedLocator(xtick_positions));
        # ax1.set_xtickslabels(xtick_lables); 
        # ax1.xaxis.set_major_formatter(FuncFormatter())
        plt.yticks(ytick_positions, ytick_lables)
        print("ytick_positions is ", ytick_positions)
        print("ytick_lables is ", ytick_lables)
        
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis('tight');
    ax1.set_xlabel(xlabel, fontsize=FontSize[0])
    ax1.set_ylabel(ylabel, fontsize=FontSize[1])
    
    ax1.tick_params(axis='x', labelsize=fontsize[0])
    ax1.tick_params(axis='y', labelsize=fontsize[1])
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    if colorbar:  
        cbar = plt.colorbar(label=colorbar_label, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=legend_size[1])
        cbar.ax.set_ylabel(colorbar_label, fontsize=legend_size[0]) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-2, 2) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.1E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
     
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(legend_size[2])
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".jpg", dpi=eps_dpi)
    else:
        plt.savefig(output_name, dpi=300)
 
    if not pltshow:
        plt.close();
        

def format_tick(tick):
    if tick>0:
        exponent = int(np.floor(np.log10(float(tick))))
    elif tick<0:
        exponent =  int(np.floor(np.log10(float(-tick))))
    else:
        exponent = 0;
    coeff   = round(tick / 10**exponent, 2)
    
    if exponent == 0:
        return f"{coeff:g}"
    else:
        return f"{coeff:g} $\\times$ 10$^{{{exponent:d}}}$"

def imshow2_old(input_array, x1beg=0, x1end=0, d1num=0, x2beg=0, x2end=0, d2num=0, xlabel="x (grid point)", ylabel="z (grid point)", vmin=0,  vmax=0, colorbar=True, colorbar_label="Relative amplitude", cmap="seismic", figsize=(10,5), output_name="tmp.eps", eps_dpi=300, title="", xtop=True, pltshow=False):
    """
    input_array:input_arry, numpy array
    pltshow: two options: True: plotshow and save figure, False: only save    
    cmap: color for imshow, "gist_rainbow", 'gray', "gist_rainbow", "seismic", "summer"
    eps_dpi: 
    output_name: output filename. if the end of filename an eps file, we will output both eps and  jpg;Otherwise, we only output jpg or others. 
    example1: P.imshow1(in_arr_kxkz, x1beg=-kx_max, x1end=+kx_max, d1num=0, x2beg=+kz_max, x2end=-kz_max, d2num=0, xlabel="Horizontal wavenumber (km$^{-1}$)", ylabel="Vertical wavenumber (km$^{-1}$)", vmin=clip1, vmax=clip2, colorbar=legend, colorbar_label=units, cmap=color, figsize=figsize1, output_name=output_name, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    example2: P.imshow1(input_arr.T, x1beg=x1beg, x1end=x1end, d1num=0, x2beg=z1end, x2end=z1beg, d2num=0, xlabel=xlabel, ylabel=ylabel, vmin=vmin,  vmax=vmax, colorbar=True, colorbar_label=colorbar_label, cmap=color, figsize=figsize, output_name=eps_name + fig_type, eps_dpi=eps_dpi, title="", xtop=True, pltshow=False);
    
    """
    if vmin==0 and vmax==0:
        vmin = input_array.min()
        vmax = input_array.max()
    if x1beg ==0 and x1end ==0 and x2beg ==0 and x2end ==0:
        x1end = input_array.shape[1]
        x2beg = input_array.shape[0]
    
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(10,5))

    
    # Plot the 2D array in the left subplot
    ax1   = fig.add_subplot( )
    
    im1   = plt.imshow(input_array, vmin=vmin, vmax=vmax, cmap=cmap, extent = (x1beg, x1end, x2beg, x2end), origin='upper');
    
    if d1num!=0 and d2num!=0:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(d1num))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(d2num))
    else:
        ax1.xaxis.set_major_locator(ticker.AutoLocator())
        ax1.yaxis.set_major_locator(ticker.AutoLocator())
        
    if xtop:
        ax1.xaxis.set_label_position("top");
        ax1.xaxis.tick_top();
        
    if title:
        plt.title(title);
    
    plt.axis('tight');
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    # Remove the margins of the subplot
    fig.set_constrained_layout(True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    
    if colorbar:
        cbar = plt.colorbar(label=colorbar_label, pad=0.01, shrink=0.7)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_ylabel(colorbar_label, fontsize=8) 
        
        print("ini cbar.get_ticks() is", cbar.get_ticks())
        
        # Set the tick formatter to a scalar formatter with 3 decimal places
        tick_format = ticker.ScalarFormatter(useMathText=True)
        tick_format.set_powerlimits( (-1, 1) )
        tick_format.set_scientific(True)
        tick_format.set_useOffset(False)
        tick_format.format = '%.2E'
        cbar.ax.yaxis.set_major_formatter(tick_format)
        
        
        # Set the number of ticks on the colorbar to 5
        # num_ticks = 5
        # ticks = ticker.LinearLocator(num_ticks).tick_values(vmin, vmax)
        # cbar.set_ticks(ticks)
        
        print("final cbar.get_ticks() is", cbar.get_ticks())
        
        
        ##set the scale size and position of colorbar
        cbar_box = cbar.ax.get_position();
        print("cbar_box is", cbar_box)
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_size(6)
        offset_text.set_verticalalignment('center')
        offset_text.set_horizontalalignment('center')
        print("offset_text.set_position is", offset_text.set_position)
        
        # canvas = fig.canvas
        # # create Agg
        # width, height = canvas.get_width_height()
        # dpi = canvas.get_renderer().dpi
        # renderer = agg.RendererAgg( width, height, dpi )
        
        # # get tex box
        # offset_text_bb = offset_text.get_window_extent(renderer=renderer)
        # offset_text_width  = offset_text_bb.width
        # offset_text_height = offset_text_bb.height
        
        # x = cbar_box.x0 + cbar_box.width + offset_text_width
        # y = cbar_box.y1 + offset_text_bb.height
        # offset_text.set_position((x, y))
        # print("offset_text_height is", offset_text_height)
        # print("offset_text_width is", offset_text_width)
        # print("offset_text.set_position is", offset_text.set_position)
    
    name = output_name[-3:]
    if name=='eps':
        plt.savefig(output_name, dpi=eps_dpi)
        plt.savefig(output_name + ".jpg", dpi=eps_dpi)
        plt.close();
    else:
        plt.savefig(output_name, dpi=300)
        plt.close();
    
    if not pltshow:
        plt.close();
            

def add_fixed_snr_noise(signal, snr):
    # Step 1: Calculate the signal power
    P = np.mean(np.square(signal))
    
    # Step 2: Convert SNR from dB to linear scale
    snr_lin = 10**(snr/10)
    
    # Step 3: Calculate noise power
    N = P / snr_lin
    
    # Step 4: Generate random noise
    noise = np.random.randn(*signal.shape)
    
    # Step 5: Normalize the noise
    noise *= np.sqrt(N)
    
    # Step 6: Add noise to signal
    noisy_signal = signal + noise
    
    return noisy_signal, noise

def estimate_snr(input_arr1,input_arr2):
    
    P_signal=np.sum(abs(input_arr1)**2);
    
    P_d=np.sum(abs(input_arr2-input_arr1)**2);
    
    SNR=10.0*np.log10(1.0*P_signal/P_d);

    return SNR
    
    
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    
    return np.random.randn(len(x)) * np.sqrt(npower)

def NCC_2d(in_arr1, in_arr2):
    
    (nx, nz)=in_arr1.shape
    cc = 0.0;
    for ix in range(0, nx):
        input0       = in_arr1[ix,:]
        input2       = in_arr2[ix,:]
        cc2          = np.dot(input0, input2)/np.sqrt(np.dot(input0, input0))/np.sqrt(np.dot(input2, input2))
        
        cc = cc + cc2;
    return 1.0*cc/nx     

def wgn2(x, snr):
    P_signal = np.sum(abs(x)**2)/len(x)
    P_noise = P_signal/10**(snr/10.0)
    
    return np.random.randn(len(x)) * np.sqrt(P_noise)

def print_numpy_array_info(in_arr, file):
    file1="\n{} shape is {}".format(file, in_arr.shape); print(file1);
    file1="{} max is {} id is {}".format(file, np.max(in_arr), np.argmax(in_arr)); print(file1);
    file1="{} max abs is {} id is {}".format(file, np.max( np.abs(in_arr)), np.argmax( np.abs(in_arr))); print(file1);
    file1="{} min is {} id is {}".format(file, np.min( (in_arr)), np.argmin( (in_arr))); print(file1);
    file1="{} min abs is {} id is {}".format(file, np.min( np.abs(in_arr)), np.argmin( np.abs(in_arr))); print(file1);
    file1="{} mean is {}".format(file, np.mean( in_arr)); print(file1);
    file1="{} std is {}".format(file, np.std( in_arr)); print(file1);
    
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
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # ######################
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
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

def fwrite_file_3d(file_name, x_data, nx=1, ny=1, nz=1, output_type="wb+"):
    
    shape = x_data.shape
    
    nz = shape[0]
    ny = shape[1]
    nx = shape[2]
    
    t_start = time.perf_counter();
#    f = open(file_name, "wb+")
    f = open(file_name, output_type)
    for ix in range(0, nx):
        for iy in range(0, ny):
            for iz in range(0, nz):
                data_float=x_data[iz][iy][ix]
                data=struct.pack("f", data_float)
                f.write(data)
    f.close();
    t_end = time.perf_counter();
    print(file_name + " has been fwrite_file_3d to np.array s)" + str((t_end - t_start)))


def fread_file_3d(file_name,x_data,nx,ny,nz,fseek_number=0):
    
    #x_data = np.zeros((nz,ny,nx));
    
    t_start = time.perf_counter();
    f = open(file_name, "rb+")
    f.seek(fseek_number*4, 0)
    for ix in range(0, nx):
        for iy in range(0, ny):
            for iz in range(0, nz):
                data = f.read(4)
                (data_float,) = struct.unpack("f", data)
                # print(data)
                # print(data_float)
                x_data[iz][iy][ix] = data_float
    f.close();
    t_end = time.perf_counter();
    print(file_name + " has been fread_file_3d to np.array s)" + str((t_end - t_start)))
    
    #return x_data

def fwrite_file_2d(file_name, x_data, nx=1, nz=1, output_type="wb+"):
    
    [nz, nx] = x_data.shape
    t_start = time.perf_counter();    
#    f = open(file_name, "wb+")
    f = open(file_name, output_type)
    for ix in range(0, nx):        
        for iz in range(0, nz):
            data_float=x_data[iz][ix]
            data=struct.pack("f", data_float)
            f.write(data)
    f.close();
    t_end = time.perf_counter();
    print(file_name + " has been fwrite_file_2d to np.array s)" + str((t_end - t_start)))


def fread_file_2d(file_name,x_data,nx,nz,fseek_number=0):
    
    t_start = time.perf_counter();    
    #x_data = np.zeros((nz,nx));

    f = open(file_name, "rb+")
    f.seek(fseek_number*4, 0)
    for ix in range(0, nx):       
        for iz in range(0, nz):
            data = f.read(4)
            (data_float,) = struct.unpack("f", data)
            # print(data)
            # print(data_float)
            x_data[iz][ix] = data_float
    f.close();
    t_end = time.perf_counter();
    print(file_name + " has been fread_file_2d to np.array s)" + str((t_end - t_start)))
    
    #return x_data

def fwrite_file_1d(file_name, x_data, nz=1, output_type="wb+"):
    
    shape = x_data.shape
    
    nz = shape[0]
    
    t_start = time.perf_counter();    
#    f = open(file_name, "wb+")
    f = open(file_name, output_type)
    for iz in range(0, nz):        
        data_float=x_data[iz]
        data=struct.pack("f", data_float)
        f.write(data)
    f.close();
    t_end = time.perf_counter();
#    print(file_name + " has been fwrite_file_1d to np.array s)" + str((t_end - t_start)))



def fread_file_1d(file_name,x_data,nz,fseek_number=0):
    
    t_start = time.perf_counter();    
    #x_data = np.zeros((nz));

    f = open(file_name, "rb+")
    f.seek(fseek_number*4, 0)
    for iz in range(0, nz):       
        data = f.read(4)
        (data_float,) = struct.unpack("f", data)
        # print(data)
        # print(data_float)
        x_data[iz] = data_float
    f.close();
    t_end = time.perf_counter();
    print(file_name + " has been fread_file_1d to np.array s)" + str((t_end - t_start)))
    
    #return x_data
