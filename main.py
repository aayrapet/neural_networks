import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from nnModules import NN_Modules
from layers import Layer
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


class MLP_Classifier(NN_Modules):

    _first_time: bool = True

    def __init__(
        self,
        nn_infra: Tuple[Layer],
        alpha: float = 0.01,
        thr: float = 1e-5,
        max_iter: int = 1000,
        batch_size: Optional[int] = None,
        seed: int = 123,
        verbose: bool = True,
        optim: str = "vanilla SGD",
        nb_epochs_early_stopping: int = 1000,
    ) -> None:
        """

        MLP
        -----

        The loss we minimize is cross-entropy (i.e., the negative log-likelihood),
        which differs from the notation used in the PDF, thus signs are different in gradients, regularisations, updates


        """
        super().__init__(optim=optim)

        
        # Parameter checks
        super().check_init_params("alpha0", alpha, float, "model parameter")
        super().check_init_params("thr", thr, float, "model parameter")
        super().check_init_params("max_iter", max_iter, int, "model parameter")
        super().check_init_params(
            "nb_epochs_early_stopping", nb_epochs_early_stopping, int, "model parameter"
        )

        if optim not in self.OPTIM_METHODS.keys():
            super().suggest_alternative(optim, self.OPTIM_METHODS.keys(), -1)

        self.nb_layers, self.nb_cnn_layers, self.network = super().check_and_build_layers(
            nn_infra
        )

        self.thr = thr
        self.max_iter = max_iter
        self.seed = seed
        self.batch_size = batch_size
        self.verbose = verbose
        self.optim = optim
        self.nb_epochs_early_stopping = nb_epochs_early_stopping
        self.alpha = alpha
        self.alpha_batch_norm_ema = 0.9
        self.model_not_trained = True

        # in dropout param regul is keep probability

        if MLP_Classifier._first_time:
            print(
                "Don't forget to normalise input data and think about Batch normalisations"
            )

            MLP_Classifier._first_time = False

  
    @staticmethod
    def verif_train_params(func: Callable) -> Callable:
        def wrapper(
            self,
            X: pd.DataFrame,
            Y: pd.DataFrame,
            X_test: Optional[pd.DataFrame] = None,
            Y_test: Optional[pd.DataFrame] = None,
            fct: Callable = accuracy_score,
        ):
            if not isinstance(X, pd.DataFrame) or not isinstance(Y, pd.DataFrame):
                raise ValueError(
                    "Input Matrix X and target matrix/vector Y have to be DataFrame matrices"
                )

            if X.shape[0] != Y.shape[0]:
                raise ValueError(
                    "Matrix X and vector/matrix Y must have same number of observations (N)"
                )
            if not np.issubdtype(X.values.dtype, np.number) or not np.issubdtype(
                Y.values.dtype, np.number
            ):
                raise ValueError(
                    "Matrix X and vector/matrix Y must have numeric values only, in your matrices somewhere i found non numeric value, so pre-process it "
                )

            if (X_test is None or Y_test is None) and not (
                X_test is None and Y_test is None
            ):

                raise ValueError(
                    "X_test or Y_test is missing but one of them is defined"
                )
            if not (X_test is None and Y_test is None):
                if not isinstance(X_test, pd.DataFrame) or not isinstance(
                    Y_test, pd.DataFrame
                ):
                    raise ValueError("X_test, Y_test have to be pandas dataframes ")
                if X_test.shape[1] != X.shape[1]:
                    raise ValueError(
                        "X train and X test have to have same nb of features "
                    )

            return func(self, X, Y, X_test, Y_test, fct)

        return wrapper

    @staticmethod
    def verif_test_params(func: Callable) -> Callable:
        def wrapper(self, X: pd.DataFrame):
            if self.model_not_trained:
                raise ValueError("Model has not been trained, so cannot predict")
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input Matrix X has to be DataFrame matrix")

            if X.shape[1] != self.p:
                raise ValueError(
                    "Matrix X_train and X_test have to have same number of columns  "
                )

            if not np.issubdtype(X.values.dtype, np.number):
                raise ValueError(
                    "Matrix X  must have numeric values only, in your matrices somewhere i found non numeric value, so pre-process it "
                )
            return func(self, X)

        return wrapper

    @verif_train_params
    def train(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        Y_test: Optional[pd.DataFrame] = None,
        fct: Callable = accuracy_score,
    ) -> None:

        self.type = "multi" if Y.shape[1] > 1 else "binary"
        self.X, self.Y = np.array(X), np.array(Y)
        self.p, self.Yncol, self.N = X.shape[1], Y.shape[1], X.shape[0]

        if self.batch_size is None:
            self.batch_size = max(100, int(self.N / 1000))  # for example

        # -------------INITIALISATION OF PARAMETERS------------------
        # B is dict of weigths
        # c is dict of biases
        # gamma  is scale parameter in batchnormalistion
        # beta is shift aprameter in batchnormalisaiton
        self.B, self.c = {}, {}
        self.gamma, self.beta = {}, {}
        # initialise weights and biases
        self.weight_init()

        # Z is linear combination Z=XB=c
        # H is non linearity applied to Z : u(Z), where u is activation fucntion
        # M is binary mask used in dropout
        # Zhat is standartised Z s.t. Zhat=(Z-mean(Z))/std(Z) at layer l
        # mean is mean(Z)
        # sigma is std(Z)
        # BN is scaled and shifted Zhat s.t BN(Z) = gamma * Zhat+ beta at layer l
        self.Z, self.H, self.M = {}, {}, {}
        self.Zhat = {}
        self.mean = {}
        self.sigma2 = {}
        self.BN = {}

        # grad_B is gradient of weight matrix at layer l
        # grad_c is gradient of bias vector at layer l
        # E is intermediate matrix for layer l (see pdf)
        # V is intermediate matrix in case of batchnorm for layer l (see pdf)
        self.E, self.V, self.grad_B, self.grad_c = {}, {}, {}, {}
        self.grad_gamma, self.grad_beta = {}, {}
        # for EMA with mean and sigma2, the final running statistics will be used for inference
        self.mean_running, self.sigma2_running = {}, {}
        # initialise parameters for EMA for some optimisers

        super().initialise_params_for_optim_algos(self.c,self.B,self.a,self.b)
        self.initialise_params_ema_batchnorm()

        # ----------------BACKPROPAGATION and SGD-------------------------------
        self.optim_algo(X_test, Y_test, fct)
        self.model_not_trained=False
        # end of function, model is trained

    def initialise_params_ema_batchnorm(self):

        self.mean_running = {
            key: np.zeros_like(val)
            for key, val in self.c.items()
            if self.network[key]["batchnorm"]
        }
        self.sigma2_running = {
            key: np.ones_like(val)
            for key, val in self.c.items()
            if self.network[key]["batchnorm"]
        }



    @verif_test_params
    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        X_test_array = np.array(X_test)
        predicted = self.forward_pass(X_test_array, None, "test")

        return self.predict_here_and_now(predicted)

    def predict_here_and_now(self, final_layer: np.ndarray) -> np.ndarray:
        if self.type == "binary":
            return np.where(final_layer > 0.5, 1, 0)
        else:
            return np.where(
                final_layer == np.max(final_layer, axis=1, keepdims=True), 1, 0
            )

    def optim_algo(
        self,
        X_test: Optional[pd.DataFrame],
        Y_test: Optional[pd.DataFrame],
        fct: Callable,
    ) -> None:

        alphaT = self.alpha * 0.01  # alphaT is 1 prct of alpha0
        T_for_alpha = 300  # for example
        k = 1
        # the xtest ytest are None or pandas df(both)
        if X_test is not None:
            X_test = np.array(X_test)
            Y_test = np.array(Y_test)

        loss_old, curr_min__loss_val = float("inf"), float("inf")
        counter_early_stop = 1

        oldalpha = None
        t = 1  # t is used in adam optim
        for epoch in range(self.max_iter):
            # SGD with random permutation at each epoch
            indices = np.random.permutation(self.N)
            X = self.X[indices]
            Y = self.Y[indices]
            start_index = 0

            while start_index < self.N:

                end_index = min(start_index + self.batch_size, self.N)

                X_batch = X[start_index:end_index]
                Y_batch = Y[start_index:end_index]

                # start with new minibatch : forward-> backward->update weights->forward(to calculate loss)
                self.forward_pass(X_batch, Y_batch, "train")
                self.calculate_gradients_backprop(X_batch, end_index - start_index)

                if k < T_for_alpha:
                    alphat = (1 - (k / T_for_alpha)) * self.alpha + (
                        k / T_for_alpha
                    ) * alphaT
                    oldalpha = alphat
                else:
                    alphat = oldalpha

                self.update_gradients(t, alphat)
                start_index = end_index
                t = t + 1

            # loss comparison new vs old (on training set)
            k = k + 1
            self.forward_pass(self.X, self.Y, "train")
            loss_new = self.loss(self.Y,self.y_hat)

            if np.abs(loss_new - loss_old) < self.thr:
                if self.verbose:
                    print(
                        f"Model terminated successfully, Converged at {epoch+1} epoch, for a given alpha :  {self.alpha} and given threshold : {self.thr} "
                    )
                return
            loss_old = loss_new

            # see how metrics evaluate over time (train set and test set (if defined))
            if self.verbose:
                if epoch % 50 == 0:

                    print(
                        "-------------------------------------------------------------------------"
                    )
                    y_predicted_train = self.predict_here_and_now(self.y_hat)
                    print(
                        f"iteration {epoch} : TRAIN {fct.__name__}  : {fct(y_predicted_train,self.Y)}, loss : {loss_new}"
                    )

            if X_test is not None:

                self.forward_pass(X_test, Y_test, "train")
                test_loss = self.loss(Y_test,self.y_hat)

                if test_loss < curr_min__loss_val:
                    curr_min__loss_val = test_loss
                    counter_early_stop = 1
                else:
                    counter_early_stop = counter_early_stop + 1

                if self.verbose:
                    if epoch % 50 == 0:

                        y_predicted_test = self.predict_here_and_now(self.y_hat)
                        print(
                            f"iteration {epoch} : TEST {fct.__name__}  : {fct(y_predicted_test,Y_test)}, loss : {test_loss}"
                        )

                if counter_early_stop >= self.nb_epochs_early_stopping:
                    print(f"early stopping at epoch {epoch}")
                    return

        print(
            f"Model terminated successfully, Did not Converge at {epoch+1} epoch, for a given alpha :  {self.alpha} and given threshold : {self.thr} "
        )

    def weight_init(
        self,
    ):
        np.random.seed(self.seed)
        for l in range(1, self.nb_layers + 1):

            if l == 1:
                self.B[l] = super().generate_weights(
                    self.p,
                    self.network[l]["nb_neurons"],
                    self.network[l]["init"],
                    self.network[l]["law"],
                )
            else:
                self.B[l] = super().generate_weights(
                    self.network[l - 1]["nb_neurons"],
                    self.network[l]["nb_neurons"],
                    self.network[l]["init"],
                    self.network[l]["law"],
                )

            if self.network[l]["batchnorm"]:
                self.gamma[l] = np.ones((1, self.network[l]["nb_neurons"]))
                self.beta[l] = np.zeros((1, self.network[l]["nb_neurons"]))

            self.c[l] = np.zeros((1, self.network[l]["nb_neurons"]))
        # set tiny variance without possibility to change for output layer
        self.a = np.random.normal(
            loc=0, scale=0.01, size=(self.network[l]["nb_neurons"], self.Yncol)
        )

        self.b = np.zeros((1, self.Yncol))


    def forward_pass(self, X, Y, train_or_test):

        for l in range(1, self.nb_layers + 1):
            if l == 1:
                self.Z[l] = X @ self.B[l] + self.c[l]
            else:
                self.Z[l] = self.H[l - 1] @ self.B[l] + self.c[l]

            if self.network[l]["batchnorm"]:
                if train_or_test == "train":

                    mean = np.mean(self.Z[l], axis=0)
                    sigma2 = np.var(self.Z[l], axis=0)
                    self.mean[l] = mean
                    self.sigma2[l] = sigma2
                else:
                    mean = self.mean_running[l]
                    sigma2 = self.sigma2_running[l]

                self.Zhat[l] = (self.Z[l] - mean) / np.sqrt((sigma2) + 1e-08)
                self.BN[l] = self.Zhat[l] * self.gamma[l] + self.beta[l]
                self.H[l] = super().u(self.BN[l], self.network[l]["activ_fct"])
            else:

                self.H[l] = super().u(self.Z[l], self.network[l]["activ_fct"])

            if train_or_test == "train":
                if self.network[l]["regul"] == "dropout":
                    self.M[l] = np.where(
                        np.random.uniform(0, 1, size=self.Z[l].shape)
                        < self.network[l]["regul_param"],
                        1,
                        0,
                    )
                    self.H[l] = self.H[l] * self.M[l] / self.network[l]["regul_param"]

        self.y_hat = self.OUTPUT_FUNCTION[self.type](
            self.H[self.nb_layers] @ self.a + self.b
        )
        if train_or_test == "test":
            return self.y_hat
        self.delta = self.y_hat - Y

    def update_gradients(self, t, alphat):

        for l in range(self.nb_layers, 0, -1):
            self.B[l] = self.B[l] - alphat * super().optim_method(
                self.grad_B[l], l, t, "v_B", "m_B"
            )
            self.c[l] = self.c[l] - alphat * super().optim_method(
                self.grad_c[l], l, t, "v_c", "m_c"
            )
            if self.network[l]["batchnorm"]:

                self.gamma[l] = self.gamma[l] - alphat * super().optim_method(
                    self.grad_gamma[l], l, t, "v_gamma", "m_gamma"
                )
                self.beta[l] = self.beta[l] - alphat * super().optim_method(
                    self.grad_beta[l], l, t, "v_beta", "m_beta"
                )
        # for convention here  l=0 then it is output layer
        self.a = self.a - alphat * super().optim_method(self.grad_a, 0, t, "v_a", "m_a")
        self.b = self.b - alphat * super().optim_method(self.grad_b, 0, t, "v_b", "m_b")

    def calculate_gradients_backprop(self, X, nb_observations_inside_batch):

        vector_ones = np.ones((1, nb_observations_inside_batch))

        for l in range(self.nb_layers, 0, -1):

            # calculate interm results before parameter gradients
            # choice to have all calculations in one function to make more readable and less messy
            if self.network[l]["batchnorm"]:
                if l == self.nb_layers:
                    V_l = (
                        self.delta
                        @ self.a.T
                        * super().uʹ(self.BN[l], self.network[l]["activ_fct"])
                        * (1 / nb_observations_inside_batch)
                    )
                else:
                    V_l = (self.E[l + 1] @ self.B[l + 1].T) * super().uʹ(
                        self.BN[l], self.network[l]["activ_fct"]
                    )

                D_l = V_l * self.gamma[l]
                m1_l = vector_ones @ D_l * (1 / nb_observations_inside_batch)
                m2_l = (
                    vector_ones
                    @ (D_l * self.Zhat[l])
                    * (1 / nb_observations_inside_batch)
                )

                self.E[l] = (D_l - m1_l - self.Zhat[l] * m2_l) / np.sqrt(
                    (self.sigma2[l]) + 1e-08
                )
                # gradeints for scale and schift
                self.grad_gamma[l] = vector_ones @ (V_l * self.Zhat[l])
                self.grad_beta[l] = vector_ones @ V_l

                # EMA update of running mean and variance
                # (done here, not in forward_pass, to ensure we only update on training steps)

                self.mean_running[l] = (
                    self.alpha_batch_norm_ema * self.mean_running[l]
                    + (1 - self.alpha_batch_norm_ema) * self.mean[l]
                )
                self.sigma2_running[l] = (
                    self.alpha_batch_norm_ema * self.sigma2_running[l]
                    + (1 - self.alpha_batch_norm_ema) * self.sigma2[l]
                )

            else:

                if l == self.nb_layers:

                    self.E[l] = (
                        self.delta
                        @ self.a.T
                        * super().uʹ(self.Z[l], self.network[l]["activ_fct"])
                        * (1 / nb_observations_inside_batch)
                    )

                else:
                    self.E[l] = (self.E[l + 1] @ self.B[l + 1].T) * super().uʹ(
                        self.Z[l], self.network[l]["activ_fct"]
                    )

            if self.network[l]["regul"] == "dropout":
                self.E[l] = self.E[l] * self.M[l] / self.network[l]["regul_param"]

            # total loss parameter gradients (sumed over i=1...N)----------
            if l > 1:
                self.grad_B[l] = self.H[l - 1].T @ self.E[l]
            else:
                self.grad_B[l] = X.T @ self.E[l]

            if self.network[l]["regul"] == "l2":
                self.grad_B[l] = (
                    self.grad_B[l] + self.network[l]["regul_param"] * self.B[l]
                )

            self.grad_c[l] = vector_ones @ self.E[l]
        self.grad_a = (
            self.H[self.nb_layers].T @ self.delta * (1 / nb_observations_inside_batch)
        )
        self.grad_b = vector_ones @ self.delta * (1 / nb_observations_inside_batch)

    def loss(self, Y,y_hat):
        return self.LOSS_FUNCTIONS[self.type](Y,y_hat)

   