from layers import ConvLayer,MaxPoolLayer,Layer
from main import MLP_Classifier

class CNN(MLP_Classifier):
    def __init__(self, nn_infra, alpha=0.01, thr=1e-5, max_iter=1000,batch_size=None,seed=123,verbose=True,optim="adam",nb_epochs_early_stopping=500):
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

    # def forward_cnn(X,Y,train_or_test):





