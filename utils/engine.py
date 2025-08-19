import torch.nn.functional as F
import numpy as np
import torch


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list =[
        [[1,0,0],[0,-1,0],[0,0,0]], 
        [[0,1,0],[0,-1,0],[0,0,0]], 
        [[0,0,1],[0,-1,0],[0,0,0]],
        [[0,0,0],[1,-1,0],[0,0,0]], 
        [[0,0,0],[0,-1,1],[0,0,0]],
        [[0,0,0],[0,-1,0],[1,0,0]], 
        [[0,0,0],[0,-1,0],[0,1,0]], 
        [[0,0,0],[0,-1,0],[0,0,1]]
    ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)
    
    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)
    
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv
    
    return contrast_depth