from numpy_cnn_operations import conv3D, MaxPooling3D,flatten_reshape3D

class Layer:
    def __init__(
        self,
        nb_neurons: int,
        activation_function: str,
        regul,
        batchnorm: bool = False,
        initial: str = "lecun",
        law: str = "normal",
    ) -> None:

        self.batchnorm=batchnorm
        self.nb_neurons = nb_neurons
        self.activation_function = activation_function
        self.initial = initial
        self.law = law
        self.regul=regul
        self.regularisation=None
        self.parameter=None
 

class ConvLayer:
    def __init__(self,in_channels,output_channels,kernel_size,stride,padding,activation_function,initial="lecun",law="normal"):

        self.in_channels=in_channels
        self.output_channels=output_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding

        self.initial=initial
        self.activation_function = activation_function
        self.law = law
        self.function=conv3D

class MaxPoolLayer:
    def __init__(self,kernel_size,stride,padding):
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.function=MaxPooling3D    

class FlatLayer:
    def __init__(self):
        self.function=flatten_reshape3D