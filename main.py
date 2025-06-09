import numpy as np
import pandas as pd


def accuracy(predicted, actual):
    return np.round(np.mean(predicted == actual), 5)


class LogisticRegression:
    def __init__(self, alpha=0.01, eps=1e-5):
        self.alpha = alpha
        self.eps = eps
        self.b = None
        self.LL = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def LL_value(self, X, b, y):
        return np.sum(
            y * np.log(self.sigmoid(X @ b)) + (1 - y) * np.log(1 - self.sigmoid(X @ b))
        )

    def gradient(self, X, b, y):
        return X.T @ (y - self.sigmoid(X @ b))

    def hessian(self, X, b, y):
        probas = self.sigmoid(X @ b) * (1 - self.sigmoid(X @ b))
        return -X.T @ np.diag(probas) @ X

    def gradient_update_ascent(self, X, y, b, alpha):
        return b + alpha * self.gradient(X, b, y)

    def gradient_ascent(self, X, y, alpha, eps, max_iter=100):
        b_old = np.zeros(((X.shape[1]), 1))
        dif = np.inf
        i = 0
        while dif > eps:
            i = i + 1
            b = self.gradient_update_ascent(X, y, b_old, alpha)
            dif = np.sum((b - b_old) ** 2)
            b_old = b
            if i > max_iter:
                print("reached max iterations, not converged")
                break

        self.b = b
        self.LL = self.LL_value(X, b, y)
        return b

    def train(self, X, y):
        y = y.reshape(-1, 1)
        self.gradient_ascent(X, y, self.alpha, self.eps)

    def predict_proba(self, X):
        return self.sigmoid(X @ self.b)

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) > thr).astype(int)


class NN_multi_layer_one_width:
    """
    correct   (verified a  lot of times)


    """

    def __init__(self, nb_layers, alpha=0.01, thr=1e-5, max_iter=500):

        if nb_layers <= 0:
            raise ValueError("nb_layers only positive integer")
        self.alpha = alpha
        self.thr = thr
        self.max_iter = max_iter
        self.nb_layers = nb_layers

    def logit_cdf(self, x):
        return 1 / (1 + np.exp(-x))

    def logit_pdf(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def initialise_coefs(self,X):
        a1 = np.random.normal(loc=0.0, scale=0.01, size=(1, 1))
        b0 = np.random.normal(loc=0.0, scale=0.01, size=(1, 1))
        b_first = np.random.normal(loc=0.0, scale=0.01, size=(X.shape[1], 1))
        epsilon_first = np.random.normal(loc=0.0, scale=0.01, size=(X.shape[0], 1))

        if self.nb_layers > 1:
            b_multi_layers = np.random.normal(
                loc=0.0, scale=0.01, size=(1, self.nb_layers - 1)
            )
            epsilon_multi_layers = np.random.normal(
                loc=0.0, scale=0.01, size=(X.shape[0], self.nb_layers - 1)
            )
            return a1, b0, b_first, epsilon_first, b_multi_layers, epsilon_multi_layers

        return a1, b0, b_first, epsilon_first, None, None

    def hidden_layer(
        self,
        index,
        X,
        b_first,
        epsilon_first,
        b_multi_layers,
        epsilon_multi_layers,
        derivative=False,
    ):

        i = 0
        while i != index:

            if i == 0:
                lin_comb = X @ b_first + epsilon_first
                h_ = self.logit_cdf(lin_comb)

            else:
                lin_comb = h_ * b_multi_layers[0][i - 1] + epsilon_multi_layers.T[
                    i - 1
                ].reshape(-1, 1)
                h_ = self.logit_cdf(lin_comb)
            i = i + 1

        return h_ if not derivative else self.logit_pdf(lin_comb)

    def probability(
        self, X, b_first, epsilon_first, b_multi_layers, epsilon_multi_layers, a1, b0
    ):
        f_xi = (
            a1
            * self.hidden_layer(
                self.nb_layers,
                X,
                b_first,
                epsilon_first,
                b_multi_layers,
                epsilon_multi_layers,
                derivative=False,
            )
            + b0
        )
        return self.logit_cdf(f_xi)

    def gradients(
        self, y, X, b_first, epsilon_first, b_multi_layers, epsilon_multi_layers, a1, b0
    ):

        gradient_b_first_storage = np.zeros(
            (X.shape[1], 1)
        )  # first b is in  Rd,  others in R
        gradient_b_multi_storage = np.zeros((1, self.nb_layers - 1))
        gradient_epsilon_all_storage = np.zeros((X.shape[0], self.nb_layers))

        gradient_a1 = self.hidden_layer(
            self.nb_layers,
            X,
            b_first,
            epsilon_first,
            b_multi_layers,
            epsilon_multi_layers,
            derivative=False,
        )
        gradient_b0 = 1
        probability = self.probability(
            X, b_first, epsilon_first, b_multi_layers, epsilon_multi_layers, a1, b0
        )

        for i in range(1, self.nb_layers + 1):  # from 1 to J layers
            # vectors multiplication element by element
            product_phi_s_prime = np.multiply.reduce(
                np.array(
                    [
                        self.hidden_layer(
                            j,
                            X,
                            b_first,
                            epsilon_first,
                            b_multi_layers,
                            epsilon_multi_layers,
                            derivative=True,
                        )
                        for j in range(i, self.nb_layers + 1)
                    ]
                )
            )
            result = np.array(
                [b_multi_layers[0][j - 1 - i] for j in range(i + 1, self.nb_layers + 1)]
            )
            if result.size == 0:
                result = np.array([1])
            # scalars multiplication
            product_b_s = np.prod(result)

            if i == 1:
                hidden_previous_layer = X
            else:
                hidden_previous_layer = self.hidden_layer(
                    i - 1,
                    X,
                    b_first,
                    epsilon_first,
                    b_multi_layers,
                    epsilon_multi_layers,
                    derivative=False,
                )

            gradient_b = np.sum(
                (y - probability)
                * a1
                * product_phi_s_prime
                * product_b_s
                * hidden_previous_layer,
                axis=0,
            )
            gradient_epsilon = (
                (y - probability) * a1 * product_phi_s_prime * product_b_s
            )
            if i == 1:
                gradient_b = gradient_b.reshape(-1, 1)
                gradient_b_first_storage = gradient_b
            else:
                gradient_b_multi_storage[:, i - 2] = gradient_b
            gradient_epsilon_all_storage[:, i - 1] = gradient_epsilon.flatten()
        gradient_a1_storage = np.sum((y - probability) * gradient_a1, axis=0)
        gradient_b0_storage = np.sum((y - probability) * gradient_b0, axis=0)

        #     print(gradient_epsilon_all_storage.shape,"storage")

        return (
            gradient_b_first_storage,
            gradient_b_multi_storage,
            gradient_epsilon_all_storage,
            gradient_a1_storage,
            gradient_b0_storage,
        )

    def gradient_update_ascent(
        self,
        y,
        X,
        b_first,
        epsilon_first,
        b_multi_layers,
        epsilon_multi_layers,
        a1,
        b0,
        alpha,
    ):

        (
            gradient_b_first_storage,
            gradient_b_multi_storage,
            gradient_epsilon_all_storage,
            gradient_a1_storage,
            gradient_b0_storage,
        ) = self.gradients(
            y, X, b_first, epsilon_first, b_multi_layers, epsilon_multi_layers, a1, b0
        )

        update_b0 = b0 + alpha * gradient_b0_storage

        update_a1 = a1 + alpha * gradient_a1_storage

        update_b_first = b_first + alpha * gradient_b_first_storage

        update_epsilon_first = epsilon_first + alpha * gradient_epsilon_all_storage[
            :, 0
        ].reshape(-1, 1)
        # print((gradient_epsilon_all_storage[:,0]).shape,"first vector di")
        # print(update_epsilon_first.shape,"updat")

        if self.nb_layers > 1:
            update_epsilon_multi = (
                epsilon_multi_layers + alpha * gradient_epsilon_all_storage[:, 1:]
            )
            update_b_multi = b_multi_layers + alpha * gradient_b_multi_storage
            return (
                update_a1,
                update_b0,
                update_b_first,
                update_epsilon_first,
                update_b_multi,
                update_epsilon_multi,
            )
        return update_a1, update_b0, update_b_first, update_epsilon_first, None, None

    def adjust_vector_dim(self, epsilon_multi, b_multi):
        for_teta_eps_mult = None
        for_teta_b_mult = None
        if epsilon_multi is not None:

            for_teta_eps_mult = epsilon_multi.flatten(order="F").reshape(-1, 1)
            for_teta_b_mult = b_multi[0].reshape(-1, 1)
        return for_teta_eps_mult, for_teta_b_mult

    def get_theta_no_None(self, *args):
        arrays = list(args)
        arrays_to_concat = [arr for arr in arrays if arr is not None]
        teta_old = np.concatenate(arrays_to_concat)
        return teta_old

    def gradient_ascent(self, X, y, alpha):

        (
            a1_old,
            b0_old,
            b_first_old,
            epsilon_first_old,
            b_multi_old,
            epsilon_multi_old,
        ) = self.initialise_coefs(X)

        for_teta_eps_mult_old, for_teta_b_mult_old = self.adjust_vector_dim(
            epsilon_multi_old, b_multi_old
        )

        teta_old = self.get_theta_no_None(
            a1_old,
            b0_old,
            b_first_old,
            epsilon_first_old,
            for_teta_b_mult_old,
            for_teta_eps_mult_old,
        )
        dif = np.inf
        i = 0
        while dif > self.thr:
            i = i + 1
            #####correct below as nor correct order, finish code and try to look for the errors then chatgpt correct, then generate dataset titanic, excecute it on logit and nn11 and try to excecute it on this
            a1, b0, b_first, epsilon_first, b_multi, epsilon_multi = (
                self.gradient_update_ascent(
                    y,
                    X,
                    b_first_old,
                    epsilon_first_old,
                    b_multi_old,
                    epsilon_multi_old,
                    a1_old,
                    b0_old,
                    alpha,
                )
            )

            for_teta_eps_mult, for_teta_b_mult = self.adjust_vector_dim(
                epsilon_multi, b_multi
            )

            teta = self.get_theta_no_None(
                a1, b0, b_first, epsilon_first, for_teta_b_mult, for_teta_eps_mult
            )
            dif = np.linalg.norm(teta - teta_old)

            teta_old = teta
            (
                a1_old,
                b0_old,
                b_first_old,
                epsilon_first_old,
                b_multi_old,
                epsilon_multi_old,
            ) = (a1, b0, b_first, epsilon_first, b_multi, epsilon_multi)
            if i > self.max_iter:
                print("reached max iterations, not converged")
                break
        print(i)
        self.a1 = a1
        self.b0 = b0
        self.b_first = b_first
        self.epsilon_first = epsilon_first
        self.b_multi = b_multi
        self.epsilon_multi = epsilon_multi

    def train(self, X, y):
        y = y.reshape(-1, 1)
        self.gradient_ascent(X, y, self.alpha)

    def predict_proba(self, X):
        # probability(self,X,b_first,epsilon_first,b_multi_layers,epsilon_multi_layers,a1,b0):
        return self.probability(
            X,
            self.b_first,
            self.epsilon_first,
            self.b_multi,
            self.epsilon_multi,
            self.a1,
            self.b0,
        )

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) > thr).astype(int)
