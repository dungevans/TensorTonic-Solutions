import numpy as np
import torch
import torch.nn as nn

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    
    x_tensor = torch.from_numpy(x).float()
    x_tensor = x_tensor.permute(0, 3, 1, 2)
    
    B, C, H, W = x_tensor.shape

   
    conv1 = nn.Conv2d(in_channels=C, out_channels=out_channels, kernel_size=3, padding=0)
    conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0)
    
    act = nn.ReLU()
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    x = act(conv1(x_tensor))
    
    skip_out = act(conv2(x)) 
    
    
    pool_out = maxpool(skip_out)
    pool_out = pool_out.permute(0, 2,3,1)
    skip_out = skip_out.permute(0, 2,3,1)
    
    return pool_out, skip_out