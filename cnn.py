from layers import ConvLayer,MaxPoolLayer,Layer,FlatLayer
from main import MLP_Classifier
import numpy as np
from sklearn.metrics import accuracy_score
from numpy_cnn_operations import *
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


class CNN(MLP_Classifier):
    def __init__(self, nn_infra, alpha=0.01, thr=1e-5, max_iter=1000,batch_size=None,seed=123,verbose=True,nb_epochs_early_stopping=500):
        self.nn_infra = nn_infra
        
   
        super().__init__(
            nn_infra=nn_infra,
            alpha=alpha,
            thr=thr,
            max_iter=max_iter,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
            optim="vanilla SGD",
            nb_epochs_early_stopping=nb_epochs_early_stopping
        )

    def forward_cnn(self,X,train_or_test):

        input_matrix=X
        
        # if train_or_test=="train":
        #                 print(0,"original image  shape:",input_matrix.shape)
        #let CV be a dictionnary with conv, MP dict with maxpool, ACV be a dict with convolutions on which activ fct was applied
        #let kernels,cvbiases be dict for conv
        #let RESCNN be result of cnn
        last=0
        for l in range(1,self.nb_cnn_layers+1):

            if self.network[l]["layer_type"].__name__=="ConvLayer":

                    #pading
                    input_matrix,pad_tuple=padding_f(input_matrix,padding=self.network[l]["padding"],filter_f=self.network[l]["kernel_size"],stride=self.network[l]["stride"])

                    if train_or_test=="train":
                        self.network[l]["pad_tuple"]=pad_tuple
                        self.PadX[l]=input_matrix

                    self.CV[l]=conv3D(input_matrix,kernel=self.kernels[l] ,bias=self.cvbiases[l])
                    self.ACV[l]=self.ACTIV_FUNCTIONS[self.network[l]["activ_fct"]](self.CV[l])
                    input_matrix=self.ACV[l]
                    # if train_or_test=="train":
                    #     print(l,"conv layer shape:",input_matrix.shape)


            elif self.network[l]["layer_type"].__name__=="MaxPoolLayer":

                self.network[l]["shape_input_matrix"]=input_matrix.shape
                
                if input_matrix.shape[0]!=1:
                    mask,out=MaxPooling3D(input_matrix,  filter_f= self.network[l]["kernel_size"],stride=self.network[l]["stride"])


                    if train_or_test=="train":
                        
                        self.maskMP[l]=mask
                    self.MP[l]=out
                    input_matrix=self.MP[l]
                    last=l
                else:
                    #ATTENTION I DONT HANDLE THIS SCENARIO IN BACKWARD SO NEVER NEVER reach the situation when maxpooling is used on matrices of size 1*1
                    #so u can stop earlier or at that exact moment and instantly go to flatter layer
                    #valid pooling has to have valid sizes first
                    raise RuntimeError("Invalid CNN design: maxpool on 1x1 tensor reached.")
                    
                # if train_or_test=="train":
                #         print(l,"maxpool layer shape:",input_matrix.shape)
                

            elif self.network[l]["layer_type"].__name__=="FlatLayer":
                original_shape,RESCNN=flatten_reshape3D(input_matrix)
                
                if train_or_test=="train":
                        # print(l,"flatten layer shape:",input_matrix.shape)
                        self.network[l]["original_shape"]=original_shape

        return RESCNN

    def predict(self,X):
        #ATTENTION IMAGE HAS TO BE OF SIZE  (H,W,C,N)!!!!!!!!(height,wiidth,nb of channels(rgb), nb of images, nothing else!!!)
        result_cnn=self.forward_cnn(X,"test")
        predicted = super().forward_pass(result_cnn,None,"test")
        return super().predict_here_and_now(predicted)

    def calculate_gradients_backprop_cnn(self):

            

            #before calculating gradients cnnn we calculate gradients mlp
            #we calculate dL/dX for cnn where X is input matrix of MLP (reshaped result of cnn)
            #dL/dX=E[1]@B[1].T 
            #but since for cnn we start at 1 and end at nb_cnn_layers+nb_layers:
            #becomes dL/dX=E[nb_cnn_layers+1]@B[nb_cnn_layers+1].T =: dL_dX
            dL_dX=self.E[self.nb_cnn_layers+1]@self.B[self.nb_cnn_layers+1].T

            grad=None
            grad_wrt_ACV=None
            for l in range(self.nb_cnn_layers, 0, -1):
                print(l,self.network[l]["layer_type"].__name__)
                
                #i suppose that at l==self.nb_cnn_layers we have flattenlayer (logic,but i admit that future error validation can be useful )
                if l==self.nb_cnn_layers:
                    grad=flatten_backward(dL_dX, self.network[l]["original_shape"])
                elif self.network[l]["layer_type"].__name__=="MaxPoolLayer":


                    grad_wrt_ACV=maxpool_backward_general(grad,self.maskMP[l],self.network[l]["shape_input_matrix"],self.network[l]["kernel_size"],self.network[l]["stride"])

                elif self.network[l]["layer_type"].__name__=="ConvLayer":
                    grad_wrt_CV=grad_wrt_ACV*super().uʹ(self.CV[l], self.network[l]["activ_fct"])
                    #let dCV_dkernel,dCV_dbias be gradients dicts
                    #stride is always 1 for conv in my setting 
                    
                    self.dCV_dkernel[l]=conv_weight_grad(self.PadX[l],grad_wrt_CV,self.network[l]["kernel_size"],1)
                    self.dCV_dbias[l]=conv_bias_grad(grad_wrt_CV)

                    #gradient of convolution result wrt input matrix X

                    dCV_dX=conv_input_grad(grad_wrt_CV,self.kernels[l],self.network[l]["pad_tuple"])
                    
                    grad=dCV_dX



    def train(
        self,
        X,
        Y,
        X_test = None,
        Y_test = None,
        fct  = accuracy_score


    ) -> None:

        # -------------INITIALISATION OF PARAMETERS CNN------------------
        self.kernels={}
        self.cvbiases={}
        self.CV={}
        self.ACV={}
        self.MP={}
        self.kernels_init()
        #save padded matrices
        self.PadX={}
        #indeces of max values within windows for maxpoll layer
        self.maskMP={}

        #store gradients wrt to kernels and biases in convolution layers 
        self.dCV_dkernel={}
        self.dCV_dbias={}

        dummy_image=np.zeros((X.shape[0],X.shape[1],X.shape[2],1))#simulate image in order to get self.p for MLP 
        dummy_result=self.forward_cnn(dummy_image,"test")
        print("dummy res shape",dummy_result.shape)

        self.type = "multi" if Y.shape[1] > 1 else "binary"
        self.X, self.Y = X,Y
        self.Yncol, self.N =  Y.shape[1], X.shape[3]
        self.p=dummy_result.shape[1]

        if self.batch_size is None:
             self.batch_size = max(100, int(self.N / 1000))  # for example

        # -------------INITIALISATION OF PARAMETERS EXTENDED MLP------------------
        # B is dict of weigths
        # c is dict of biases
        # gamma  is scale parameter in batchnormalistion (mlp here)
        # beta is shift aprameter in batchnormalisaiton
        self.B, self.c = {}, {}
        self.gamma, self.beta = {}, {}
        # initialise weights and biases
        super().weight_init()
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
        # E is intermediate matrix for layer l (see neural_networks.pdf)
        # V is intermediate matrix in case of batchnorm for layer l (see pdf)
        self.E, self.V, self.grad_B, self.grad_c = {}, {}, {}, {}
        self.grad_gamma, self.grad_beta = {}, {}
        # for EMA with mean and sigma2, the final running statistics will be used for inference
        self.mean_running, self.sigma2_running = {}, {}
        # initialise parameters for EMA for some optimisers
        super().initialise_params_for_optim_algos(self.c,self.B,self.a,self.b)
        super().initialise_params_ema_batchnorm()

        ----------------BACKPROPAGATION and SGD-------------------------------
        self.optim_algo(X_test, Y_test, fct)
        self.model_not_trained=False
        end of function, model is trained



    def kernels_init(
        self,
    ):
        np.random.seed(self.seed)
        for l in range(1, self.nb_cnn_layers + 1):

            if self.network[l]["layer_type"].__name__=="ConvLayer":
                self.kernels[l] = super().generate_weights(
                        fan_in=self.network[l]["in_channels"]*(self.network[l]["kernel_size"]**2),
                        fan_out=self.network[l]["output_channels"]*(self.network[l]["kernel_size"]**2),
                        
                        shape=(self.network[l]["kernel_size"],self.network[l]["kernel_size"],self.network[l]["in_channels"],self.network[l]["output_channels"]),
                        
                        init=self.network[l]["init"],
                        law=self.network[l]["law"],
                     
                    )
                self.cvbiases[l] = np.zeros((self.network[l]["output_channels"]))
     
    def update_gradients_cnn(self, t, alphat):

        for l in range(1, self.nb_cnn_layers+1):

            if self.network[l]["layer_type"].__name__=="ConvLayer":

                self.kernels[l] = self.kernels[l] - alphat * self.dCV_dkernel[l]
                self.cvbiases[l] = self.cvbiases[l] - alphat * self.dCV_dbias[l]

    def test(self,X,Y):

        RESCNN=self.forward_cnn(X,"train")
        super().forward_pass(RESCNN,Y,"train")
        super().calculate_gradients_backprop(RESCNN, RESCNN.shape[0])
        self.calculate_gradients_backprop_cnn()

    
    def optim_algo(
        self,
        X_test = None,
        Y_test = None,
        fct: Callable,
    ) -> None:

        alphaT = self.alpha * 0.01  # alphaT is 1 prct of alpha0
        T_for_alpha = 300  # for example
        k = 1
        

        loss_old, curr_min__loss_val = float("inf"), float("inf")
        counter_early_stop = 1

        oldalpha = None
        t = 1  # t is used in adam optim
        for epoch in range(self.max_iter):
            # SGD with random permutation at each epoch
            indices = np.random.permutation(self.N)
            X = self.X[:,:,:,indices]
            Y = self.Y[indices]
            start_index = 0

            while start_index < self.N:

                end_index = min(start_index + self.batch_size, self.N)

                X_batch = X[:,:,:,start_index:end_index]
                Y_batch = Y[start_index:end_index]

                # start with new minibatch : forward-> backward->update weights->forward(to calculate loss)
                
                RESCNN=self.forward_cnn(X_batch,"train")
                super().forward_pass(RESCNN, Y_batch, "train")
                super().calculate_gradients_backprop(RESCNN, end_index - start_index)
                self.calculate_gradients_backprop_cnn()


                if k < T_for_alpha:
                    alphat = (1 - (k / T_for_alpha)) * self.alpha + (
                        k / T_for_alpha
                    ) * alphaT
                    oldalpha = alphat
                else:
                    alphat = oldalpha

                super().update_gradients(t, alphat)
                self.update_gradients_cnn(None,alphat)

                start_index = end_index
                t = t + 1

            # loss comparison new vs old (on training set)
            #i can do also by minibatches and then average losses but i dont have time 
            k = k + 1
            RESCNN=self.forward_cnn(self.X,"test")#here i do on all dataset(not efficient)
            super().forward_pass(RESCNN, self.Y, "train")#not clean but it works only with this 
            loss_new = super().loss(self.Y,self.y_hat)

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
                    y_predicted_train = super().predict_here_and_now(self.y_hat)
                    print(
                        f"iteration {epoch} : TRAIN {fct.__name__}  : {fct(y_predicted_train,self.Y)}, loss : {loss_new}"
                    )

            if X_test is not None:
                #do same but also on xtest

                RESCNN=self.forward_cnn(X_test,"test")#here i do on all dataset(not efficient)
                super().forward_pass(RESCNN, Y_test, "train")#not clean but it works only with this                 
                test_loss = super().loss(Y_test,self.y_hat)

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




