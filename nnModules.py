
import difflib
from layers import ConvLayer,MaxPoolLayer,Layer,FlatLayer
import numpy as np 
from typing import  (
    Any,
    Callable,
    Dict,
    List,
    KeysView,
    Optional,
    Tuple,
    Union,
)


class NN_Modules:
    def __init__(self,optim):

        if optim!= "vanilla SGD":
            self.beta1=0.9
            if optim == "adam":
                self.beta2=0.99

        self.optim=optim



        self.OUTPUT_FUNCTION: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "multi": self.__softmax_output,
            "binary": self.__sigmoid_output,
        }

        self.ACTIV_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "sigmoid": self.__sigmoid,
            "relu": self.__relu,
            "tanh": self.__tanh,
        }

        self.ACTIV_DERIV_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "sigmoid": self.__sigmoidʹ,
            "relu": self.__reluʹ,
            "tanh": self.__tanhʹ,
        }

        self.INIT_FUNCTIONS: Dict[str, Callable[[int, int, str], np.ndarray]] = {
            "lecun": self.__lecun,
            "xavier": self.__xavier,
            "he": self.__he,
            "random": self.__random,
        }

        self.LOSS_FUNCTIONS: Dict[str, Callable[[np.ndarray], float]] = {
            "multi": self.cross_entropy_multi,
            "binary": self.cross_entropy_binary,
        }


        self.OPTIM_METHODS: Dict[
            str, Callable[[np.ndarray, int, int, str, str], np.ndarray]
        ] = {
            "vanilla SGD": self.__vanillaSGD,  # classic st gradient descent
            "momentum": self.__momentum,
            "adam": self.__adam,
            "rmsprop": self.__rmsprop,
        }
        self.REGUL_METHODS: Dict[str, None] = {"l2": None, "dropout": None}
        self.LAWS: Dict[str, None] = {"normal": None, "uniform": None}

    def __lecun(self, fan_in, fan_out, law):
        if law == "normal":
            return np.random.normal(0, np.sqrt(1 / fan_in), (fan_in, fan_out))
        else:
            a = np.sqrt(3 / fan_in)
            return np.random.uniform(-a, a, (fan_in, fan_out))

    def __xavier(self, fan_in, fan_out, law):
        if law == "normal":
            return np.random.normal(
                0, np.sqrt(2 / (fan_in + fan_out)), (fan_in, fan_out)
            )
        else:
            a = np.sqrt(6 / (fan_in + fan_out))
            return np.random.uniform(-a, a, (fan_in, fan_out))

    def __he(self, fan_in, fan_out, law):
        if law == "normal":
            return np.random.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
        else:
            a = np.sqrt(6 / fan_in)
            return np.random.uniform(-a, a, (fan_in, fan_out))

    def __random(self, fan_in, fan_out, law):
        if law == "normal":
            return np.random.normal(loc=0, scale=1, size=(fan_in, fan_out))
        elif law == "uniform":
            return np.random.uniform(low=-1, high=1, size=(fan_in, fan_out))

    def u(self, x, type_activation):
        # in future need to import function from another library so not to define inside MLP
        return self.ACTIV_FUNCTIONS[type_activation](x)

    def uʹ(self, x, type_activation):
        return self.ACTIV_DERIV_FUNCTIONS[type_activation](x)

    def __sigmoid(self, x):

        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def __sigmoidʹ(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def __relu(self, x):
        return np.where(x <= 0, 0, x)

    def __reluʹ(self, x):
        return np.where(x <= 0, 0, 1)

    def __tanh(self, x):
        return np.tanh(x)

    def __tanhʹ(self, x):
        return 1 - (np.tanh(x)) ** 2

    def __softmax_output(self, x):
        # https://en.wikipedia.org/wiki/Softmax_function
        x_shift = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x_shift)
        return ex / np.sum(ex, axis=1, keepdims=True)

    def __sigmoid_output(self, x):
        return self.__sigmoid(x)
    
    def cross_entropy_binary(self, Y, y_hat):
        eps = 1e-8
        p = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    def cross_entropy_multi(self, Y, y_hat):
        eps = 1e-8
        p = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(np.sum(Y * np.log(p), axis=1))

    def generate_weights(self, fan_in, fan_out, init, law):
        return self.INIT_FUNCTIONS[init](fan_in, fan_out, law)
    
    def __vanillaSGD(self, gradient, layer_index, t, *args):
        return gradient

    def __momentum(self, gradient, layer_index, t, *args):

        this_attrib = getattr(self, args[0])  # supports only one (v)

        this_attrib[layer_index] = (
            1 - self.beta1
        ) * gradient + self.beta1 * this_attrib[layer_index]
        return this_attrib[layer_index]

    def __adam(self, gradient, layer_index, t, *args):

        v_attr = getattr(self, args[0])
        m_attr = getattr(self, args[1])

        v = (1 - self.beta1) * gradient + self.beta1 * v_attr[layer_index]
        m = (1 - self.beta2) * (gradient**2) + self.beta2 * m_attr[layer_index]
        v_attr[layer_index] = v
        m_attr[layer_index] = m

        vhat = v / (1 - self.beta1**t)
        mhat = m / (1 - self.beta2**t)

        return vhat / (np.sqrt(mhat) + 1e-8)

    def __rmsprop(self, gradient, layer_index, t, *args):

        this_attrib = getattr(self, args[0])  # supports only one (v)

        v = (1 - self.beta1) * (gradient**2) + self.beta1 * this_attrib[layer_index]
        this_attrib[layer_index] = v

        return gradient / (np.sqrt(v) + 1e-8)



    def suggest_alternative(self,target: str, keys_allowed, i: int) -> None:
        if target.lower() in keys_allowed:
            # allow only exact notation, no undercase
            msg = f"Layer {i+1}: Unknown '{target}'. " f"Did you mean '{target.lower()}'?"
            raise ValueError(msg)

        try:
            suggestion = difflib.get_close_matches(
                target, list(keys_allowed), n=1, cutoff=0.2
            )
        except Exception as e:
            msg = (
                f"Layer {i+1}: Unknown '{target}' and no close match found. "
                f"(Error: {e})"
            )
            raise ValueError(msg) from e

        if suggestion:
            msg = f"Layer {i+1}: Unknown '{target}'. " f"Did you mean '{suggestion[0]}'?"
            raise ValueError(msg)

        raise ValueError(f"Layer {i+1}: Unknown '{target}' and no close match found.")


    def check_init_params(self,
        str_param: str, param, type_of_param: type, prefix: str
    ) -> None:
        if not isinstance(param, type_of_param):
            msg = (
                f"{prefix}: {str_param} has to be {type_of_param.__name__}, "
                f"but declared as {param.__class__.__name__}"
            )
            raise ValueError(msg)
        if type_of_param == float or type_of_param == int:
            if param <= 0:
                msg = (
                    f"{prefix}: {str_param} has to have positive non zero float, "
                    f"but we have {str_param} =  {param}"
                )
                raise ValueError(msg)


    def check_and_build_layers(self, layers):


            result = {}
            if layers is None or len(layers) == 0 or not isinstance(layers, tuple):
                raise ValueError("You have to define at least one layer in a tuple")

            mlp_layers_present=False
            cnn_layers_present=False
            flat_layer_present=False
            cnn_layers=0
            for i, layer in enumerate(layers):

                #check order of layers: 

                if not isinstance(layer,Layer) and not isinstance(layer,ConvLayer) and not isinstance(layer,MaxPoolLayer) and not isinstance(layer,FlatLayer):
                    raise ValueError("You specified non existing layer")

                elif (isinstance(layer,ConvLayer) or isinstance(layer,MaxPoolLayer) or isinstance(layer,FlatLayer))  and mlp_layers_present:
                    raise ValueError("cnn layer can't be after mlp layer")
                elif  isinstance(layer,FlatLayer) and not cnn_layers_present:
                    raise ValueError("flat layer can't be before cnn layer")

                #check parameters 
                if isinstance(layer,Layer) or isinstance(layer,ConvLayer):
                    self.check_init_params("batchnorm", layer.batchnorm, bool, f"{layer.__class__.__name__} [{i+1}]")
                if isinstance(layer,Layer):
                    self.check_init_params("nb of neurons", layer.nb_neurons, int, f"Layer [{i+1}]")

                if isinstance(layer,Layer) or isinstance(layer,ConvLayer):
                    self.check_init_params("initialisation", layer.initial, str, f"{layer.__class__.__name__} [{i+1}]")

                if isinstance(layer,Layer) or isinstance(layer,ConvLayer):
                    if not isinstance(layer.regul, tuple) and layer.regul != None:
                        raise ValueError(
                            f"{layer.__class__.__name__} [{i+1}] : regul has to be tuple or None, but is {layer.regul.__class__.__name__}"
                        )

                    try:
                        layer.regularisation, layer.parameter = (
                            layer.regul if layer.regul else (None, None)
                        )
                    except Exception :
                        raise ValueError(
                            f"{layer.__class__.__name__} [{i+1}] : regul has to contain nothing or 2 parameters: regularisation and corresponding parameter"
                        )

                    if layer.regul:
                        self.check_init_params(
                            "regularisation", layer.regularisation, str, f"{layer.__class__.__name__} [{i+1}]"
                        )
                        self.check_init_params(
                            "param regul", layer.parameter, float, f"{layer.__class__.__name__} [{i+1}]"
                        )

                
                    self.check_init_params(
                        "activation function", layer.activation_function, str, f"{layer.__class__.__name__} [{i+1}]"
                    )
                    self.check_init_params("law", layer.law, str, f"{layer.__class__.__name__} [{i+1}]")

                    verif_keys = [
                        layer.activation_function,
                        layer.initial,
                        layer.regularisation,
                        layer.law,
                    ]
                    possible_lists = [
                        self.ACTIV_FUNCTIONS.keys(),
                        self.INIT_FUNCTIONS.keys(),
                        self.REGUL_METHODS.keys(),
                        self.LAWS.keys(),
                    ]

                    ##verify that for each layers strings we have known string parameters---------------
                    for x, y in zip(verif_keys, possible_lists):

                        if x not in y and x != None:

                            self.suggest_alternative(x, y, i)

                #check parameters inherent to cnn
                if isinstance(layer,ConvLayer):
                    self.check_init_params("in_channels", layer.in_channels, int, f"ConvLayer [{i+1}]")
                    self.check_init_params("output_channels", layer.output_channels, int, f"ConvLayer [{i+1}]")
                if isinstance(layer,ConvLayer) or isinstance(layer,MaxPoolLayer) :
                    self.check_init_params("kernel_size", layer.kernel_size, int, f"{layer.__class__.__name__} [{i+1}]")
                    self.check_init_params("stride", layer.stride, int, f"{layer.__class__.__name__} [{i+1}]")
                    self.check_init_params("padding", layer.padding, bool, f"{layer.__class__.__name__} [{i+1}]")

                # construct json of architecture
                if isinstance(layer,Layer):
                    result[i + 1] = {
                        "layer_type" : Layer,
                        "nb_neurons": layer.nb_neurons,
                        "activ_fct": layer.activation_function,
                        "regul": layer.regularisation,
                        "regul_param": layer.parameter,
                        "init": layer.initial,
                        "law": layer.law,
                        "batchnorm": layer.batchnorm,
                    }
                    mlp_layers_present=True
                elif isinstance(layer,ConvLayer):

                    result[i + 1] = {
                        "layer_type" : ConvLayer,
                        "in_channels": layer.in_channels,
                        "output_channels": layer.output_channels,
                        "kernel_size": layer.kernel_size,
                        "stride": layer.stride,
                        "padding": layer.padding,
                        "activ_fct": layer.activation_function,
                        "regul": layer.regularisation,
                        "regul_param": layer.parameter,
                        "init": layer.initial,
                        "law": layer.law,
                        "batchnorm": layer.batchnorm,
                        "fct" : layer.function
                    }
                    cnn_layers_present=True
                    cnn_layers=cnn_layers+1
                elif isinstance(layer,MaxPoolLayer):

                    result[i + 1] = {
                        "layer_type" : MaxPoolLayer,
                        "kernel_size": layer.kernel_size,
                        "stride": layer.stride,
                        "padding": layer.padding,
                        "fct" : layer.function
                    }
                    cnn_layers_present=True
                    cnn_layers=cnn_layers+1
                elif  isinstance(layer,FlatLayer):


                    result[i + 1] = {
                        
                        "layer_type" : FlatLayer,
                        "fct" : layer.function
                    }

                    flat_layer_present=True
                    cnn_layers=cnn_layers+1

            nb_mlp_layers = i + 1-cnn_layers#because i need to separate them when using forward and backward pass
            return nb_mlp_layers,cnn_layers, result


    def initialise_params_for_optim_algos(self,c,B,a,b):
        vector = {key: np.zeros_like(val) for key, val in c.items()}
        for_B = {key: np.zeros_like(val) for key, val in B.items()}
        for_c = vector  # same shapes
        for_a = {}
        for_b = {}
        for_a[0] = np.zeros_like(a)
        for_b[0] = np.zeros_like(b)

        if self.optim != "vanilla SGD":
            self.v_B = for_B
            self.v_c = for_c
            self.v_a = for_a
            self.v_b = for_b
            self.v_gamma = vector
            self.v_beta = vector
            

            if self.optim == "adam":
                self.m_B = for_B
                self.m_c = for_c
                self.m_a = for_a
                self.m_b = for_b
                self.m_gamma = vector
                self.m_beta = vector
           
    @staticmethod
    def check_optim_methods(func):
        def wrapper(self, gradient, layer_index, t, *args):

            if self.optim != "vanilla SGD":
                for el in args:
                    if not hasattr(self, el):
                        raise ValueError(f"not recognised attribute '{el}'")
            return func(self, gradient, layer_index, t, *args)

        return wrapper

    @check_optim_methods
    def optim_method(self, gradient, layer_index, t, *args):

        return self.OPTIM_METHODS[self.optim](gradient, layer_index, t, *args)

    

   

