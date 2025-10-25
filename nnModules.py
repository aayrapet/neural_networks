
import difflib
from layers import ConvLayer,MaxPoolLayer,Layer,FlatLayer

def suggest_alternative(target: str, keys_allowed, i: int) -> None:
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


def check_init_params(
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


def check_and_build_layers( layers,possible_functions,possible_init,possible_regul,possible_laws) :

        result = {}
        if layers is None or len(layers) == 0 or not isinstance(layers, tuple):
            raise ValueError("You have to define at least one layer in a tuple")

        mlp_layers_present=False
        cnn_layers_present=False
        flat_layer_present=False
        for i, layer in enumerate(layers):


            



            #check order of layers: 

            if not isinstance(layer,Layer) and not isinstance(layer,ConvLayer) and not isinstance(layer,MaxPoolLayer) and not isinstance(layer,FlatLayer):
                raise ValueError("You specified non existing layer")

            elif (isinstance(layer,ConvLayer) or isinstance(layer,MaxPoolLayer) or isinstance(layer,FlatLayer))  and mlp_layers_present:
                raise ValueError("cnn layer can't be after mlp layer")
            elif  isinstance(layer,FlatLayer) and not cnn_layers_present:
                raise ValueError("flat layer can't be before cnn layer")

            if isinstance(layer,Layer) or isinstance(layer,ConvLayer):
                check_init_params("batchnorm", layer.batchnorm, bool, f"{layer.__class__.__name__} [{i+1}]")
            if isinstance(layer,Layer):
                check_init_params("nb of neurons", layer.nb_neurons, int, f"Layer [{i+1}]")

            if isinstance(layer,Layer) or isinstance(layer,ConvLayer):
                check_init_params("initialisation", layer.initial, str, f"{layer.__class__.__name__} [{i+1}]")

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
                    check_init_params(
                        "regularisation", layer.regularisation, str, f"{layer.__class__.__name__} [{i+1}]"
                    )
                    check_init_params(
                        "param regul", layer.parameter, float, f"{layer.__class__.__name__} [{i+1}]"
                    )

            
                check_init_params(
                    "activation function", layer.activation_function, str, f"{layer.__class__.__name__} [{i+1}]"
                )
                check_init_params("law", layer.law, str, f"{layer.__class__.__name__} [{i+1}]")

                verif_keys = [
                    layer.activation_function,
                    layer.initial,
                    layer.regularisation,
                    layer.law,
                ]
                possible_lists = [
                    possible_functions,
                    possible_init,
                    possible_regul,
                    possible_laws,
                ]

                ##verify that for each layers strings we have known string parameters---------------
                for x, y in zip(verif_keys, possible_lists):

                    if x not in y and x != None:

                        suggest_alternative(x, y, i)


            if isinstance(layer,ConvLayer):
                check_init_params("in_channels", layer.in_channels, int, f"ConvLayer [{i+1}]")
                check_init_params("output_channels", layer.output_channels, int, f"ConvLayer [{i+1}]")
            if isinstance(layer,ConvLayer) or isinstance(layer,MaxPoolLayer) :
                check_init_params("kernel_size", layer.kernel_size, int, f"{layer.__class__.__name__} [{i+1}]")
                check_init_params("stride", layer.stride, int, f"{layer.__class__.__name__} [{i+1}]")
                check_init_params("padding", layer.padding, bool, f"{layer.__class__.__name__} [{i+1}]")

            # construct json of architecture
            if isinstance(layer,Layer):
                result[i + 1] = {
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
            elif isinstance(layer,MaxPoolLayer):

                result[i + 1] = {
                    
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                    "fct" : layer.function
                }
                cnn_layers_present=True
            elif  isinstance(layer,FlatLayer):


                flat_layer_present=True




        nb_layers = i + 1
        return nb_layers, result