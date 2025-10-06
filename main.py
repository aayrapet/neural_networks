import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import difflib


def accuracy(pred,actual,type="binary"):

    if actual.shape[1]==1:
        return np.mean((pred==actual).astype(int))
    elif actual.shape[1]>1:
        return np.mean(np.argmax(pred,axis=1)==np.argmax(actual,axis=1))


class LogisticRegression:
    def __init__(self, alpha=0.01, eps=1e-5,max_iter=1000):
        self.alpha = alpha
        self.eps = eps
        self.b = None
        self.LL = None
        self.max_iter=max_iter

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def LL_value(self, X, b, y):
        return np.mean(
            y * np.log(self.sigmoid(X @ b)) + (1 - y) * np.log(1 - self.sigmoid(X @ b))
        )

    def gradient(self, X, b, y):
        return (X.T @ (y - self.sigmoid(X @ b)))/X.shape[0]

    def hessian(self, X, b, y):
        probas = self.sigmoid(X @ b) * (1 - self.sigmoid(X @ b))
        return -X.T @ np.diag(probas) @ X

    def gradient_update_ascent(self, X, y, b, alpha):
        return b + alpha * self.gradient(X, b, y)

    def gradient_ascent(self, X, y, alpha, eps):
        b_old = np.zeros(((X.shape[1]), 1))
        old_ll=self.LL_value(X,b_old,y)
        dif = np.inf
        i = 0
        while dif > eps:
            i = i + 1
            b = self.gradient_update_ascent(X, y, b_old, alpha)
            ll=self.LL_value(X,b,y)
            dif = np.sqrt(np.sum((b - b_old) ** 2))
            old_ll=ll
            b_old = b
            if i > self.max_iter:
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
        predictions= (self.predict_proba(X) > thr).astype(int)
        return predictions.reshape(1,-1)



class Layer:
    def __init__(self,nb_neurons,activation_function,regul=None,initial="random",law="normal"):
        
        self.nb_neurons=nb_neurons
        self.activation_function=activation_function
        self.initial=initial
        self.law=law
        self.regul=regul
def suggest_alternative(target,list_allowed,i):
   
    if target.lower() in list_allowed:
        #allow only exact notation, no undercase
        raise ValueError(
            f"Layer {i+1}: Unknown  '{target}'. Did you mean '{target.lower()}'?"
        )
    try : 
        suggestion = difflib.get_close_matches(target, list_allowed, n=1, cutoff=0.2)
    except : 
        raise ValueError(f"Layer {i+1}: Unknown '{target}' and no close match found.")
    if suggestion:
        raise ValueError(f"Layer {i+1}: Unknown '{target}'. Did you mean '{suggestion[0]}'?")
    else:
        raise ValueError(f"Layer {i+1}: Unknown '{target}' and no close match found.")
    


def check_init_params(str_param,param,type_of_param,prefix):
    if not isinstance(param, type_of_param) :
                    raise ValueError(f"{prefix}: {str_param} has to be {type_of_param.__name__} but declared as {param.__class__.__name__}")
    if type_of_param==float or type_of_param==int:
        if param <= 0:
                        raise ValueError(f"{prefix}: {str_param} has to have positive non zero float but we have {str_param} =  {param}")



class MLP_Classifier:

    _first_time=True
 
    def __init__(self,nn_infra,alpha=0.01,thr=1e-5,max_iter=1000,batch_size=None,seed=123,verbose=True,optim="vanilla SGD"):
                """

                MLP
                -----

                The loss we minimize is cross-entropy (i.e., the negative log-likelihood),
                which differs from the notation used in the PDF, thus signs are different in gradients, regularisations, updates


                """
                self.OUTPUT_FUNCTION={
                "multi" : self.__softmax_output,
                "binary" : self.__sigmoid_output,
                }

                self.ACTIV_FUNCTIONS = {
                "sigmoid" : self.__sigmoid,
                "relu" : self.__relu,
                "tanh" : self.__tanh

                }

                self.ACTIV_DERIV_FUNCTIONS = {
                    "sigmoid": self.__sigmoidʹ,
                    "relu": self.__reluʹ,
                    "tanh": self.__tanhʹ,
                }
  
                self.INIT_FUNCTIONS = {
                    "lecun": self.__lecun,
                    "xavier": self.__xavier,
                    "he": self.__he,
                    "random": self.__random,
                }
                self.LOSS_FUNCTIONS={
                    'multi' : self.__cross_entropy_multi,
                    'binary' : self.__cross_entropy_binary
                }
                self.OPTIM_METHODS={
                    "vanilla SGD" : self.__vanillaSGD,#classic st gradient descent
                    "momentum" : self.__momentum,
                    "adam" : self.__adam,
                    "rmsprop" : self.__rmsprop,
                
                }
                self.REGUL_METHODS={
                    "l2" : "self.__l2",
                    "dropout" : "self.dropout"
                }
                check_init_params('alpha',alpha,float,"model parameter" )
                check_init_params('thr',thr,float,"model parameter" )
                check_init_params('max_iter',max_iter,int,"model parameter" )

                if optim not in self.OPTIM_METHODS.keys():
                    suggest_alternative(optim,self.OPTIM_METHODS.keys(),-1)

                self.nb_layers,self.network=self.check_and_build_layers(nn_infra)
                self.alpha = alpha
                self.thr = thr
                self.max_iter=max_iter
                self.seed=seed
                self.batch_size=batch_size
                self.verbose=verbose
                self.optim=optim
                

                if MLP_Classifier._first_time:
                    print("Don't forget to normalise input data and think about Batch normalisations")
    
                    MLP_Classifier._first_time=False
    def check_and_build_layers(self,layers):

    
        possible_functions = self.ACTIV_FUNCTIONS.keys()
        possible_init=self.INIT_FUNCTIONS.keys()
        possible_regul=self.REGUL_METHODS.keys()
      
        possible_laws=["normal","uniform"]
        result={}
        for i, layer in enumerate(layers):
            ##verify layers formats---------------

            check_init_params("nb of neurons",layer.nb_neurons,int,f"Layer [{i+1}]")
            check_init_params("initialisation",layer.initial,str,f"Layer [{i+1}]")
            
            if not isinstance(layer.regul,tuple) and   layer.regul!=None:
                    raise ValueError(f"Layer [{i+1}] : regul has to be list or None, but is {layer.regul.__class__.__name__}")

            try:
                        layer.regularisation, layer.parameter = layer.regul if layer.regul else (None, None)
            except:
                        raise ValueError(f"Layer [{i+1}] : regul has to contain nothing or  2 parameters: regularisation and corresponding parameter")
                        
            if layer.regul: 
                            check_init_params("regularisation",layer.regularisation,str,f"Layer [{i+1}]")
                            check_init_params("param regul",layer.parameter,float,f"Layer [{i+1}]")
            
    
            check_init_params("activation function",layer.activation_function,str,f"Layer [{i+1}]")
            check_init_params("law",layer.law,str,f"Layer [{i+1}]")

            verif_keys=[layer.activation_function,layer.initial,layer.regularisation,layer.law]
            possible_lists=[possible_functions,possible_init,possible_regul,possible_laws]

            ##verify that for each layers strings we have known string parameters---------------
            for x,y in zip(verif_keys,possible_lists):

                if x not in  y and x!=None:
            
                    suggest_alternative(x,y,i)
            
       

            #construct json of architecture
            result[i+1]=  {
                "nb_neurons": layer.nb_neurons,
                "activ_fct": layer.activation_function,
                "regul": layer.regularisation,
                "regul_param": layer.parameter,
                "init": layer.initial,
                "law" : layer.law
            }

        nb_layers=(i+1)
        return nb_layers ,result


    @staticmethod
    def verif_train_params(func):
        def wrapper(self,X,Y):
            if not isinstance(X,pd.DataFrame) or not isinstance(Y,pd.DataFrame):
                raise ValueError("Input Matrix X and target matrix/vector Y have to be DataFrame matrices")

            if X.shape[0]!=Y.shape[0]:
                raise ValueError("Matrix X and vector/matrix Y must have same number of observations (N)")
            if not np.issubdtype(X.values.dtype, np.number)  or not np.issubdtype(Y.values.dtype, np.number):
                raise ValueError("Matrix X and vector/matrix Y must have numeric values only, in your matrices somewhere i found non numeric value, so pre-process it ")
            return func(self,X,Y)
        return wrapper 

    @staticmethod
    def verif_test_params(func):
        def wrapper(self,X):
            if not isinstance(X,pd.DataFrame):
                raise ValueError("Input Matrix X has to be DataFrame matrix")

            
            if X.shape[1]!=self.p:
                raise ValueError("Matrix X_train and X_test have to have same number of columns  ")

            
            if not np.issubdtype(X.values.dtype, np.number):
                raise ValueError("Matrix X  must have numeric values only, in your matrices somewhere i found non numeric value, so pre-process it ")
            return func(self,X)
        return wrapper 
                
    @verif_train_params
    def train(self,X,Y):
        self.type="multi"   if Y.shape[1]>1 else 'binary' 
        self.X,self.Y=np.array(X),np.array(Y)
        self.p,self.Yncol,self.N=X.shape[1],Y.shape[1],X.shape[0]
        
        #initialisation -------------------------
        self.B,self.c={},{}
        self.weight_init()
        
        #INITIALISE all H,Z for the first time -------------------------
        self.Z,self.H,self.M={},{},{}
        
        #calculate gradients via back propagation, update them via optimiser------------
        self.E, self.grad_B,self.grad_c={},{},{}


        self.initialise_params_for_optim_algos()
        self.optim_algo()

    def initialise_params_for_optim_algos(self):

        if self.optim!="vanilla SGD":
            self.v_B = [np.zeros_like(b) for b in self.B]
            self.v_c = [np.zeros_like(c) for c in self.c]
            self.v_a = np.zeros_like(self.a)
            self.v_b = np.zeros_like(self.b)
            self.beta1=0.9
            if self.optim=="adam":
                self.m_B = [np.zeros_like(b) for b in self.B]
                self.m_c = [np.zeros_like(c) for c in self.c]
                self.m_a = np.zeros_like(self.a)
                self.m_b = np.zeros_like(self.b)
                self.beta2=0.99


    @verif_test_params
    def predict(self,X_test):


        X_test_array=np.array(X_test)
        predicted=self.forward_pass(X_test_array,None,"test")

        return self.predict_here_and_now(predicted)
    
    def predict_here_and_now(self,final_layer):
        if self.type=="binary":
            return np.where(final_layer>0.5,1,0)
        else: 
            return np.where(final_layer==np.max(final_layer,axis=1, keepdims=True),1,0)

    def optim_algo(self):

        loss_old=float("inf")
        t=1
        for epoch in range(self.max_iter):

            # SGD with random permutation at each epoch 
            indices = np.random.permutation(self.N)
            X = self.X[indices]
            Y = self.Y[indices]

            start_index=0

            while start_index<self.N:

                end_index=min(start_index+self.batch_size,self.N)

                X_batch=X[start_index:end_index]
                Y_batch=Y[start_index:end_index]

                #start with new minibatch : forward-> backward->update weights->forward(to calculate loss)
                self.forward_pass(X_batch,Y_batch,"train")
                self.calculate_gradients_backprop(X_batch,end_index-start_index)
                self.update_gradients(t)
                start_index=end_index
                t=t+1
        
            self.forward_pass(self.X,self.Y,"train")
            loss_new=self.loss(self.Y)

            if self.verbose:
                if epoch%100==0:
                    
                    y_predicted=self.predict_here_and_now(self.y_hat)
                    print(f"iteration {epoch} : accuracy  : {accuracy(y_predicted,self.Y)}, loss : {loss_new}")

            if np.abs(loss_new-loss_old)<self.thr:
                if self.verbose:
                    print(f"Model terminated successfully, Converged at {epoch+1} epoch, for a given alpha :  {self.alpha} and given threshold : {self.thr} ")
                #calculate maybe also accuracies just for info (train and test sets)
                return 
            loss_old=loss_new
        print(f"Model terminated successfully, Did not Converge at {epoch+1} epoch, for a given alpha :  {self.alpha} and given threshold : {self.thr} ")
        

    def weight_init(self,):
        np.random.seed(self.seed)
        for l in range(1,self.nb_layers+1):
            
            if l == 1:
                 self.B[l] = self.__generate_weights(
                     self.p,
                     self.network[l]["nb_neurons"],
                     self.network[l]["init"],
                     self.network[l]["law"],
                 )
            else:
                 self.B[l] = self.__generate_weights(
                     self.network[l - 1]["nb_neurons"],
                     self.network[l]["nb_neurons"],
                     self.network[l]["init"],
                     self.network[l]["law"],
                 )
 

            self.c[l]=np.zeros((1,self.network[l]["nb_neurons"]))
        #set tiny variance without possibility to change for output layer 
        self.a=np.random.normal(loc=0,scale=0.01,size=(self.network[l]["nb_neurons"],self.Yncol))

        self.b=np.zeros((1,self.Yncol))
       
    def forward_pass(self,X,Y,train_or_test):

        for l in range(1,self.nb_layers+1):
            if l==1:
                self.Z[l]=X@self.B[l]+self.c[l] 
            else:
                self.Z[l]=self.H[l-1]@self.B[l]+self.c[l]
            self.H[l]=self.u(self.Z[l],self.network[l]["activ_fct"])
            if train_or_test=="train":
                if self.network[l]["regul"]=="dropout":
                    self.M[l]=np.where(np.random.uniform(0,1,size=self.Z[l].shape)<self.network[l]["regul_param"],1,0)
                    self.H[l]=self.H[l]*self.M[l]/self.network[l]["regul_param"]

        self.y_hat=self.OUTPUT_FUNCTION[self.type](self.H[self.nb_layers]@self.a+self.b)
        if train_or_test=="test":
            return self.y_hat
        self.delta=(self.y_hat-Y)
     
    def update_gradients(self,t):



        for l in range(self.nb_layers,0,-1):
            self.B[l]=self.B[l]-self.alpha*self.optim_method(self.grad_B[l],l,"weight",t)
            self.c[l]=self.c[l]-self.alpha*self.optim_method(self.grad_c[l],l,"bias",t)

        #for convention here  l=0 then it is output layer 
        self.a=self.a-self.alpha*self.optim_method(self.grad_a,0,"weight",t)
        self.b=self.b-self.alpha*self.optim_method(self.grad_b,0,"bias",t)

    def optim_method(self,gradient,layer_index,type,t):

        return self.OPTIM_METHODS[self.optim](gradient,layer_index,type,t)


    def __vanillaSGD(self,gradient,layer_index,type,t):
        return gradient

    def __momentum(self,gradient,layer_index,type,t):


        if layer_index>0:

            if type=="weight":
           
                self.v_B[layer_index-1]=(1-self.beta1)*gradient+self.beta1*self.v_B[layer_index-1]
                return self.v_B[layer_index-1]
            elif type=="bias":
                self.v_c[layer_index-1]=(1-self.beta1)*gradient+self.beta1*self.v_c[layer_index-1]
                return self.v_c[layer_index-1]
        else:
            if type=="weight":
           
                self.v_a=(1-self.beta1)*gradient+self.beta1*self.v_a
                return self.v_a
            elif type=="bias":
                self.v_b=(1-self.beta1)*gradient+self.beta1*self.v_b
                return self.v_b

    def __adam(self,gradient,layer_index,type,t):

        if layer_index>0:

            if type=="weight":
           
                v=(1-self.beta1)*gradient+self.beta1*self.v_B[layer_index-1]
                m=(1-self.beta2)*(gradient**2)+self.beta2*self.m_B[layer_index-1]
                self.v_B[layer_index-1]=v
                self.m_B[layer_index-1]=m

            elif type=="bias":
                v=(1-self.beta1)*gradient+self.beta1*self.v_c[layer_index-1]
                m=(1-self.beta2)*(gradient**2)+self.beta2*self.m_c[layer_index-1]
                self.v_c[layer_index-1]=v
                self.m_c[layer_index-1]=m

        else:
            if type=="weight":
           
                v=(1-self.beta1)*gradient+self.beta1*self.v_a

                m=(1-self.beta2)*(gradient**2)+self.beta2*self.m_a
                self.v_a=v
                self.m_a=m
            elif type=="bias":
                v=(1-self.beta1)*gradient+self.beta1*self.v_b

                m=(1-self.beta2)*(gradient**2)+self.beta2*self.m_b
                self.v_b=v
                self.m_b=m
        
        vhat=v/(1-self.beta1**t)
        mhat=m/(1-self.beta2**t)
               
        return vhat/(np.sqrt(mhat)+1e-8)


    def __rmsprop(self,gradient,layer_index,type,t):

        if layer_index>0:

            if type=="weight":
                v=(1-self.beta1)*(gradient**2)+self.beta1*self.v_B[layer_index-1]
                self.v_B[layer_index-1]=v
               
            elif type=="bias":
                v=(1-self.beta1)*(gradient**2)+self.beta1*self.v_c[layer_index-1]
                self.v_c[layer_index-1]=v
        else:
            if type=="weight":
           
                v=(1-self.beta1)*(gradient**2)+self.beta1*self.v_a
                self.v_a=v
            elif type=="bias":
                v=(1-self.beta1)*(gradient**2)+self.beta1*self.v_b
                self.v_b=v
        return gradient/(np.sqrt(v)+1e-8)


      
    def calculate_gradients_backprop(self,X,nb_observations_inside_batch):

        for l in range(self.nb_layers,0,-1):

            if l==self.nb_layers:
                # if self.type=="binary":
                #     self.a=self.a.reshape(1,-1)
                self.E[l]=self.delta@self.a.T*self.uʹ(self.Z[l],self.network[l]["activ_fct"])*(1/nb_observations_inside_batch)
                
            else:
                self.E[l]=(self.E[l+1]@self.B[l+1].T)*self.uʹ(self.Z[l],self.network[l]["activ_fct"])

            if self.network[l]["regul"]=="dropout":
                    self.E[l]=self.E[l]*self.M[l]/self.network[l]["regul_param"]
                
            if l>1:
                self.grad_B[l]=self.H[l-1].T@self.E[l]
            else:
                self.grad_B[l]=X.T@self.E[l]
            
            if self.network[l]["regul"]=="l2":
                    self.grad_B[l]=self.grad_B[l]+self.network[l]["regul_param"]*self.B[l]
            
            self.grad_c[l]=np.ones((1,nb_observations_inside_batch))@self.E[l]
        self.grad_a=self.H[self.nb_layers].T@self.delta*(1/nb_observations_inside_batch)
        self.grad_b=np.ones((1,nb_observations_inside_batch))@self.delta*(1/nb_observations_inside_batch)
        
    def loss(self,Y):
        return self.LOSS_FUNCTIONS[self.type](Y)

    def __cross_entropy_binary(self,Y):
        #solve numerical stability issues
        return -np.mean(Y*np.log(self.y_hat)+(1-Y)*np.log(1-self.y_hat))
    def __cross_entropy_multi(self,Y):
        return -np.mean(np.sum(Y*np.log(self.y_hat),axis=1))


    def __generate_weights(self,fan_in,fan_out,init,law):
        return self.INIT_FUNCTIONS[init](fan_in,fan_out,law)
    def __lecun(self,fan_in,fan_out,law):

        if law=="normal":
            return np.random.normal(loc=0,scale=1/fan_in,size=(fan_in,fan_out))
        elif law=="uniform":
            return np.random.uniform(low=-np.sqrt(3/fan_in),high=np.sqrt(3/fan_in),size=(fan_in,fan_out))
    def __xavier(self,fan_in,fan_out,law):

        if law=="normal":
            return np.random.normal(loc=0,scale=np.sqrt(2/(fan_in+fan_out)),size=(fan_in,fan_out))
        elif law=="uniform":
            return np.random.uniform(low=-np.sqrt(6/fan_in+fan_out),high=np.sqrt(6/fan_in+fan_out),size=(fan_in,fan_out))
    def __he(self,fan_in,fan_out,law):

        if law=="normal":
            return np.random.normal(loc=0,scale=2/(fan_in),size=(fan_in,fan_out))
        elif law=="uniform":
            return np.random.uniform(low=-np.sqrt(6/fan_in),high=np.sqrt(6/fan_in),size=(fan_in,fan_out))
    def __random(self,fan_in,fan_out,law):
        if law=="normal":
            return np.random.normal(loc=0,scale=1,size=(fan_in,fan_out))
        elif law=="uniform":
            return np.random.uniform(low=-1,high=1,size=(fan_in,fan_out))
    def u(self,x,type_activation):
        #in future need to import function from another library so not to define inside MLP
        return self.ACTIV_FUNCTIONS[type_activation](x)
    def uʹ(self,x,type_activation):
        return self.ACTIV_DERIV_FUNCTIONS[type_activation](x)
    def __sigmoid(self,x):

        return np.where(x>=0,
             1/(1+np.exp(-x))
            , np.exp(x) / (1 + np.exp(x)))
    def __sigmoidʹ(self,x):
        return self.__sigmoid(x)*(1-self.__sigmoid(x))
    def __relu(self,x):
        return np.where(x<=0,
             0
            , x)
    def __reluʹ(self,x):
        return np.where(x<=0,
             0
            , 1)
    def __tanh(self,x):
        return np.tanh(x)   
    def __tanhʹ(self,x):
        return 1 - (np.tanh(x))**2
    def __softmax_output(self,x):
        return np.exp(x)/(np.sum(np.exp(x),axis=1).reshape(-1,1))
    def __sigmoid_output(self,x):
        return self.__sigmoid(x)
        
