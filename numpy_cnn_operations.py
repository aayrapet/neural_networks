import math as mt 
import numpy as np 
import math
from numpy.lib.stride_tricks import sliding_window_view


def calculate_optimal_padding(H, W, f, s):

    # ceil and floor strictly in this order to match convolutiono padding 
    # --- Height dimension ---
    p_top = math.ceil((s * math.ceil(H / s) - H + f - s) / 2)
    p_bottom = math.floor((s * math.ceil(H / s) - H + f - s) / 2)

    # --- Width dimension ---
    p_left = math.ceil((s * math.ceil(W / s) - W + f - s) / 2)
    p_right = math.floor((s * math.ceil(W / s) - W + f - s) / 2)

    return p_top, p_bottom, p_left, p_right

def padding_f(x,padding,filter_f,stride):

    if padding: 
        H,W=x.shape[0],x.shape[1]
        p_top, p_bottom, p_left, p_right=calculate_optimal_padding(H,W,filter_f,stride)
        # print(p_top, p_bottom, p_left, p_right)
        if x.ndim==3:
            padding_width=((p_top,p_bottom),(p_left,p_right),(0,0))
        elif x.ndim==4:
            padding_width=((p_top,p_bottom),(p_left,p_right),(0,0),(0,0))

        x=np.pad(x, pad_width=padding_width, mode='constant', constant_values=0)
    return x

def convolution_one_image(x ,nb_filters,kernel,bias,padding=True):
    #let image 3D : 1 image of size H*W *nb of channels (rgd)

    x=padding_f(x,padding,len(kernel),1)#in pooling stride is 1

    #invert kernel to do convolution 
    kernel=kernel[::-1,::-1,:,:]
    rk=kernel.reshape(kernel.shape[0]*kernel.shape[1],kernel.shape[2],kernel.shape[3])
    sw=sliding_window_view(x,(len(kernel),len(kernel)),axis=(0,1))
    rsw=sw.reshape(sw.shape[0],sw.shape[1],sw.shape[2],-1)
    #convolution image*filter 
    out=np.einsum('ijkl,lkz->ijkz',rsw,rk)
    sumed_over_chanels=np.sum(out,axis=2)
    return sumed_over_chanels+bias


def convolution_multi_images(x ,kernel,bias,padding=True):
    #let image 4D : N images of size H*W *nb of channels (rgd)


    #apply padding (or not) to input image (tensor  4D)
    x=padding_f(x,padding,len(kernel),1)#in pooling stride is 1

    #invert kernel to do convolution 
    kernel=kernel[::-1,::-1,:,:]

    #need to reshape kernel matrix (H x W size) into just 1D vector to allow vector multiplication
    rk=kernel.reshape(kernel.shape[0]*kernel.shape[1],kernel.shape[2],kernel.shape[3])#kernel is symetric (H=W)

    #this is the core of convolution : instead of brute force heavy for loops (as in neural_networks.pdf)-> we use cpp function
    sw=sliding_window_view(x,(len(kernel),len(kernel)),axis=(0,1))
    rsw=sw.reshape(sw.shape[0],sw.shape[1],sw.shape[2],sw.shape[3],-1)
    #convolution image*filter 
    out=np.einsum('ijkvl,lkz->ijzvk',rsw,rk)
    sumed_over_chanels=np.sum(out,axis=4)#we need to sum over channels 
    return sumed_over_chanels+bias.reshape(1,1,len(bias),1)#add bias

def conv3D( x: np.ndarray ,kernel,bias,padding=True):

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
  
    #conditions on image to be done separately in outer functions 
    # if x.ndim>4 or x.ndim<3:
    #     raise ValueError(f"image has to be 3 dimensional or 4 dim, you have : {x.ndim}D")
    # if kernel.shape[0]!=kernel.shape[1]:
    #     raise ValueError("kernel matrices have to be symmetric")
    # if kernel.shape[0]>x.shape[0] or  kernel.shape[0]>x.shape[1]:
    #     raise ValueError("kernel dim has to be less then image dim")
    # if x.shape[2]!=3:
    #     raise ValueError(f"image has to be rgb with 3 color  channels, you have {x.shape[2]} color channels ")
    
    return conv_dim[x.ndim](x,kernel,bias,padding)
    
def MaxPooling3D(x,filter_f,stride=1,padding=True):

    shape=(filter_f,filter_f)
    x=padding_f(x,padding,filter_f,stride)

    v = sliding_window_view(x, shape,axis=(0,1))

    if x.ndim==3:
        return v[::stride, ::stride].max(axis=(3,4))#3 and 4 dim are our little matrices f*f

    elif x.ndim==4:
        return v[::stride, ::stride].max(axis=(4,5))#2 and 3 are dims of f*f matrices 
    #note that 2 last dims are our dims 
 
def flatten_reshape3D(x):
    x = np.transpose(x, (3, 0, 1, 2))   # (N,H,W,C)
    return x.reshape(x.shape[0], -1)