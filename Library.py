#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 09:32:20 2023

@author: zhangw0c
"""

##1
import sys
import os
# home_path= os.getenv("HOME"); 
# sys.path.append(home_path + "/zw_lib/python/func/")
sys.path.append("./func")
import common  as C
import plot_func as P
import torch_func as T
import image_domain_torch as I

import Net as N
# from networks import normalized_equilibrium_u_net as N_unet

import matplotlib.pyplot as plt
#import cv2


####2
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import filtfilt
from scipy.sparse.linalg import lsqr
from numba import jit

from pylops import LinearOperator
from pylops.utils                      import dottest
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.utils.tapers               import *
from pylops.basicoperators             import *
from pylops.signalprocessing           import *
from pylops.waveeqprocessing.kirchhoff import Kirchhoff
from pylops.avo.poststack              import PoststackLinearModelling, PoststackInversion

from pylops.optimization.leastsquares  import *
from pylops.optimization.sparsity      import *

from pyproximal.proximal import *
from pyproximal.optimization.primal import *
from pyproximal.optimization.primaldual import *
from pylops import TorchOperator


###3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint_sequential

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cp.cuda.Device(0).use()


##4 another self
# from ID_cuda import NonStationaryConvolve3D
# from torchoperator import TorchOperator



###5
# pytorch 设置随机种子（我试下来是有用的）(万一没用了，参考https://blog.csdn.net/weixin_40400177/article/details/105625873)
def seed_torch(seed = 10):
    random.seed(seed) # python seed
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    np.random.seed(seed) # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed) # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
    torch.backends.cudnn.enabled = True # pytorch使用CUDANN加速，即使用GPU加速
seed_value=10
seed_torch(seed = seed_value)


