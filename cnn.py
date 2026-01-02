from layers import ConvLayer,MaxPoolLayer,Layer,FlatLayer
from main import MLP_Classifier
import numpy as np
from sklearn.metrics import accuracy_score

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

    def forward_cnn(self,X):

        input_matrix=X
        print(input_matrix.shape)
        #let CV be a dictionnary with conv, MP dict with maxpool, ACV be a dict with convolutions on which activ fct was applied
        #let kernels,cvbiases be dict for conv
        #let RESCNN be result of cnn
        for l in range(1,self.nb_cnn_layers+1):

            if self.network[l]["layer_type"].__name__=="ConvLayer":
                    self.CV[l]=self.network[l]["fct"](input_matrix,kernel=self.kernels[l] ,bias=self.cvbiases[l],padding=self.network[l]["padding"])
                    self.ACV[l]=self.ACTIV_FUNCTIONS[self.network[l]["activ_fct"]](self.CV[l])
                    input_matrix=self.ACV[l]
                    print(input_matrix.shape)
            elif self.network[l]["layer_type"].__name__=="MaxPoolLayer":
                self.MP[l]=self.network[l]["fct"](input_matrix,  filter_f= self.network[l]["kernel_size"],stride=self.network[l]["stride"], padding=self.network[l]["padding"])
                input_matrix=self.MP[l]
                print(input_matrix.shape)

            elif self.network[l]["layer_type"].__name__=="FlatLayer":
                RESCNN=self.network[l]["fct"](input_matrix)
                print(input_matrix.shape)
        return RESCNN

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

        dummy_image=np.zeros((X.shape[0],X.shape[1],X.shape[2],1))#simulate image in order to get self.p for MLP 
        dummy_result=self.forward_cnn(dummy_image)

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

        # ----------------BACKPROPAGATION and SGD-------------------------------
        # self.optim_algo(X_test, Y_test, fct)
        # self.model_not_trained=False
        # end of function, model is trained



    def kernels_init(
        self,
    ):

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
     





