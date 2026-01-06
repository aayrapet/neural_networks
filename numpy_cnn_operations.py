import math as mt 
import numpy as np 
import math
from numpy.lib.stride_tricks import sliding_window_view


def calculate_optimal_padding(H, W, f, s):
    #https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
    #attenton, i do non symmetric padding 
    # ceil and floor strictly in this order to match convolutiono padding 
    # --- Height dimension ---
    p_top =  math.floor((s * math.ceil(H / s) - H + f - s) / 2) 
    p_bottom =   math.ceil((s * math.ceil(H / s) - H + f - s) / 2)

    # --- Width dimension ---
    p_left =   math.floor((s * math.ceil(W / s) - W + f - s) / 2)
    p_right =  math.ceil((s * math.ceil(W / s) - W + f - s) / 2)
    return p_top, p_bottom, p_left, p_right

def padding_f(x,padding,filter_f,stride):
    padding_width=((0,0),(0,0),(0,0),(0,0))

    if padding: 
        H,W=x.shape[0],x.shape[1]
        p_top, p_bottom, p_left, p_right=calculate_optimal_padding(H,W,filter_f,stride)
        # print(p_top, p_bottom, p_left, p_right)
        if x.ndim==3:
            padding_width=((p_top,p_bottom),(p_left,p_right),(0,0))
        elif x.ndim==4:
            padding_width=((p_top,p_bottom),(p_left,p_right),(0,0),(0,0))

        x=np.pad(x, pad_width=padding_width, mode='constant', constant_values=0)
    return x,padding_width

def convolution_one_image(x ,nb_filters,kernel,bias,padding=True):
    #let image 3D : 1 image of size H*W *nb of channels (rgd)
    #as exolained in neural_networks.pdf in a photo, i reshape kernels and do dot product between this vector and reshaped windows
    #which is done by einsum 

    x=padding_f(x,padding,len(kernel),1)#in pooling stride is 1 (in my setting )

    #invert kernel to do convolution 
    kernel=kernel[::-1,::-1,:,:]
    rk=kernel.reshape(kernel.shape[0]*kernel.shape[1],kernel.shape[2],kernel.shape[3])
    sw=sliding_window_view(x,(len(kernel),len(kernel)),axis=(0,1))
    rsw=sw.reshape(sw.shape[0],sw.shape[1],sw.shape[2],-1)
    #convolution image*filter 
    out=np.einsum('ijkl,lkz->ijkz',rsw,rk)
    sumed_over_chanels=np.sum(out,axis=2)
    return sumed_over_chanels+bias


def convolution_multi_images(x ,kernel,bias):
    #let image 4D : N images of size H*W *nb of channels (rgd)
    # stride is 1 for convolution layer (in my setting)

    #need to reshape kernel matrix (H x W size) into just 1D vector to allow vector multiplication
    rk=kernel.reshape(kernel.shape[0]*kernel.shape[1],kernel.shape[2],kernel.shape[3])#kernel is symetric (H=W)

    #this is the core of convolution : instead of brute force heavy for loops (as in neural_networks.pdf)-> we use cpp function
    sw=sliding_window_view(x,(len(kernel),len(kernel)),axis=(0,1))
    rsw=sw.reshape(sw.shape[0],sw.shape[1],sw.shape[2],sw.shape[3],-1)
    #convolution image*filter 
    out=np.einsum('ijkvl,lkz->ijzvk',rsw,rk)
    sumed_over_chanels=np.sum(out,axis=4)#we need to sum over channels 
    return sumed_over_chanels+bias.reshape(1,1,len(bias),1)#add bias


def conv3D( x: np.ndarray ,kernel,bias):

    """
    let x be one image or collection of images:


    -> if one image then the size of x is (H, W, 3) (height,width,3 color chanels)
    -> if many images then the size of x is  (H, W, 3,nunmber of images )


    for convolution stride is always 1 even if specified 

    """
    conv_dim={
    4 : convolution_multi_images,
    3 : convolution_one_image   
    }
  
    return conv_dim[x.ndim](x,kernel,bias)
    
def MaxPooling3D(x,filter_f,stride=2):
    #no padding for maxpol layer here
    shape=(filter_f,filter_f)
   
    v = sliding_window_view(x, shape, axis=(0,1))   # -> (H, W, C, N, 2, 2)
    v = v[::stride, ::stride]                                 # apply stride
    out = v.max(axis=(-1,-2))                       # (H, W, C, N)
    mask = v == out[..., None, None]  
    return mask,out#mask,maxpooled
    #note that 2 last dims are our dims 
 
def flatten_reshape3D(x):
    x = np.transpose(x, (3, 0, 1, 2))   # (N,H,W,C)
    return x.shape,x.reshape(x.shape[0], -1)


# BACKWARD PROCESSES--------------------------------------------------
def flatten_backward(grad,original_shape):
    grad = grad.reshape(original_shape)          # (N,H,W,C)
    grad = np.transpose(grad, (1,2,3,0))         # (H,W,C,N)
    return grad


def maxpool_backward_general(dout, mask, x_shape, F, S):
    # derivative dL/dx up to next layer : (H, W, C, N)
    # mask : (H, W, C, N, F, F) we look at all f*f submatrices and store its max indexes as in :
    #https://www.educative.io/answers/how-to-backpropagate-through-max-pooling-layers
    #if there are overlapping indexes then we sum them (stride<f)

    H, W, C, N = dout.shape

    dx = np.zeros(x_shape, dtype=dout.dtype)

    # (H, W, C, N, F, F)
    expanded = dout[..., None, None] * mask

    # spatial index grids
    i = np.arange(H).reshape(H, 1, 1, 1, 1, 1)
    j = np.arange(W).reshape(1, W, 1, 1, 1, 1)
    a = np.arange(F).reshape(1, 1, 1, 1, F, 1)
    b = np.arange(F).reshape(1, 1, 1, 1, 1, F)

    y = (i * S + a)              # (H,W,1,1,F,1)
    x = (j * S + b)              # (1,W,1,1,1,F)

    # C and N index grids
    c_index = np.arange(C).reshape(1,1,C,1,1,1)
    n_index = np.arange(N).reshape(1,1,1,N,1,1)

    # broadcast all to expanded shape
    y = np.broadcast_to(y, expanded.shape)
    x = np.broadcast_to(x, expanded.shape)
    c_index = np.broadcast_to(c_index, expanded.shape)
    n_index = np.broadcast_to(n_index, expanded.shape)

    # flatten everything
    y = y.ravel()
    x = x.ravel()
    c_index = c_index.ravel()
    n_index = n_index.ravel()
    expanded_flat = expanded.ravel()

    # scatter-add → correct even if overlapping
    np.add.at(dx, (y, x, c_index, n_index), expanded_flat)

    return dx

def conv_weight_grad(X, dZ, F, S):
    # X: (H, W, C_in, N)
    # dZ: (H, W, C_out, N)
    # F: filter size
    # S: stride

    H, W, C_in, N = X.shape
    _, _, C_out, _ = dZ.shape

    #Extract all input windows used in forward
    # shape -> (H, W, C_in, N, F, F)
    windows =sliding_window_view(X, (F, F), axis=(0,1))
    windows =windows[::S, ::S]  #stride           

    # reorder to align channels last for broadcasting
    # -> (H, W, F, F, C_in, N)
    windows= np.moveaxis(windows, (2,3), (4,5))

    #expand dZ to multiply each window
    # dZ: (H, W, C_out, N)
    # -> (H, W, 1, 1, 1, C_out, N)
    dZ_exp= dZ[:, :, None, None, None, :, :]

    # 3) multiply and sum over (N, H, W)
    # result -> (F, F, C_in, C_out)
    dW= (windows[..., None, :] * dZ_exp).sum(axis=(0,1, -1))

    return dW

def conv_input_grad(dZ, W,pad_tuple):
    #dL/dx=conv(pad(dl/dz),flipped(kernel))
    #https://www.youtube.com/watch?v=Pn7RK7tofPg&list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu&index=8

    pad_t, pad_b=pad_tuple[0]
    pad_l, pad_r=pad_tuple[1]
    # dZ: (H, W, C_out, N)
    # W : (F, F, C_in, C_out)
    H, W_out, C_out, N = dZ.shape
    F, _, C_in, _ = W.shape
    pad = F - 1
    #pad dZ on spatial dims
    dZ_padded = np.pad(
        dZ,
        pad_width=((pad, pad), (pad, pad), (0, 0), (0, 0)),
        mode='constant',constant_values=0
    )
    
    #sliding windows from padded dZ
    # -> (H_in, W_in, C_out, N, F, F)
    windows = sliding_window_view(dZ_padded, (F, F), axis=(0, 1))

    # reorder to put F dims together cleanly
    # -> (H_in, W_in, F, F, C_out, N)
    windows = np.moveaxis(windows, (2, 3), (4, 5))
    #flip W spatially AND swap Cin/Cout roles
    # rot180 on spatial + transpose C dims
    # -> (F, F, C_out, C_in)
    W_flip = np.flip(W, axis=(0, 1)).transpose(0, 1, 3, 2)
    #multiply and sum over (F,F,C_out)
    # result -> (H_in, W_in, C_in, N)
    dX = (windows[..., None, :] * W_flip[None, None, ... , None]).sum(axis=(2, 3, 4))

     #CROP to remove forward padding and restore original matrix before it was convolved and padded
    if pad_t or pad_b or pad_l or pad_r:
        dX = dX[pad_t : dX.shape[0] - pad_b,
                pad_l : dX.shape[1] - pad_r,
                :, :]

    return dX

def conv_bias_grad(dZ):
    # dZ: (H_out, W_out, C_out, N)
    return dZ.sum(axis=(0,1,3)) 