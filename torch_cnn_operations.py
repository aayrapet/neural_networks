from numpy_cnn_operations import calculate_optimal_padding
import torch.nn.functional as F
import torch
    

def padding_torch(img_array_perm,padding,stride,kernel_size):
        if padding: 

            #this padding is correct as can i allow assymetric padding and i use ALL pixels at borders
            H=img_array_perm[0,0,:,:].shape[0]
            W=img_array_perm[0,0,:,:].shape[1]

            p_top, p_bottom, p_left, p_right=calculate_optimal_padding(H,W,f=kernel_size,s=stride)

            img_array_perm=F.pad(img_array_perm,(p_left,p_right,p_top,p_bottom))
        return img_array_perm

def convolution_torch(img_array_perm,kernels_perm,bias,padding=True,stride=1):
    kernels_perm=torch.flip(kernels_perm, dims=[2, 3])
    kernel_size=kernels_perm.shape[2]
    img_array_perm=padding_torch(img_array_perm,padding,stride,kernel_size)
    out = F.conv2d(
        img_array_perm,          # shape (N, C_in, H, W)
        kernels_perm,    # shape (C_out, C_in, kH, kW)
        stride=stride,   
    )
    return out.add_(bias.reshape(1,len(kernels_perm),1,1))

def maxpooling_torch(img_array_perm,kernel_size,stride=2,padding=True):

    img_array_perm=padding_torch(img_array_perm,padding,stride,kernel_size)

    out = F.max_pool2d(
        img_array_perm,
        kernel_size=kernel_size,  # size of pooling window
        stride=stride,     
    )
    return out