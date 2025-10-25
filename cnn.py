from layers import ConvLayer,MaxPoolLayer,Layer

class CNN:



    def __init__(
        self,
        nn_infra
        
    ) -> None:
        """

        MLP
        -----

        The loss we minimize is cross-entropy (i.e., the negative log-likelihood),
        which differs from the notation used in the PDF, thus signs are different in gradients, regularisations, updates


        """

        self.nn_infra=nn_infra