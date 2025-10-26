from layers import ConvLayer,MaxPoolLayer,Layer
from main import MLP_Classifier
import torch
class CNN(MLP_Classifier):
    def __init__(self, nn_infra, alpha=0.01, thr=1e-5, max_iter=1000,batch_size=None,seed=123,verbose=True,optim="adam",nb_epochs_early_stopping=500,device="cpu"):
        self.nn_infra = nn_infra
        
        # Call parent constructor correctly
        super().__init__(
            nn_infra=nn_infra,
            alpha=alpha,
            thr=thr,
            max_iter=max_iter,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
            optim=optim,
            nb_epochs_early_stopping=nb_epochs_early_stopping
        )
   
        self.device=device
        if device=="cuda":
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)




    # def forward_cnn(X,Y,train_or_test):

    # def train(
    #     self,
    #     X,
    #     Y

    # ) -> None:


    #     self.conv_bias={}
    #     self.conv_kernel={}






    #     self.type = "multi" if Y.shape[1] > 1 else "binary"
    #     self.X, self.Y = X,Y
    #     self.Yncol, self.N =  Y.shape[1], X.shape[0]

    #     self.p=???


    #     if self.batch_size is None:
    #         self.batch_size = max(100, int(self.N / 1000))  # for example

    #     # -------------INITIALISATION OF PARAMETERS MLP------------------
    #     # B is dict of weigths
    #     # c is dict of biases
    #     # gamma  is scale parameter in batchnormalistion
    #     # beta is shift aprameter in batchnormalisaiton
    #     self.B, self.c = {}, {}
    #     self.gamma, self.beta = {}, {}
    #     # initialise weights and biases
    #     super().weight_init()



    def kernels_init(
        self,
    ):
        
        for l in range(1, self.nb_cnn_layers + 1):

            if self.network[l]["layer_type"].__name__=="ConvLayer":
                self.conv_kernel[l] = super().generate_weights(
                        fan_in=self.network[l]["in_channels"]*(self.network[l]["kernel_size"]**2),
                        fan_out=self.network[l]["output_channels"]*(self.network[l]["kernel_size"]**2),
                        shape=(self.network[l]["output_channels"],self.network[l]["in_channels"],self.network[l]["kernel_size"],self.network[l]["kernel_size"]),
                        init=self.network[l]["init"],
                        law=self.network[l]["law"],
                        device=self.device
                    )

                # if self.network[l]["batchnorm"]:
                #   
                self.conv_bias[l] = torch.zeros((1, self.network[l]["output_channels"],1,1))
     





