from torch_cnn_operations import convolution_torch, maxpooling_torch

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

        self.nb_neurons = nb_neurons
        self.activation_function = activation_function
        self.initial = initial
        self.law = law
        self.regul = regul
        self.batchnorm = batchnorm

class ConvLayer:
    def __init__(self,in_channels,output_channels,kernel_size,stride,padding,batchnorm,initial,activation_function,law,regul):

        self.in_channels=in_channels
        self.output_channels=output_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.batchnorm=batchnorm
        self.initial=initial
        self.activation_function = activation_function
        self.law = law
        self.regul = regul
        self.function=convolution_torch

class MaxPoolLayer:
    def __init__(self,kernel_size,stride,padding):
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.function=maxpooling_torch    



!!!!!!!!!
class FlatLayer:
    def __init__(self):
        self.function=flatten