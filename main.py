import numpy as np
import pandas as pd




#risk of overfit because of different biases for each layer so read papers or statquest 
def accuracy(pred,actual):
    return np.mean((pred==actual).astype(int))


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



class NN_multi_layer_one_width:
    """
    problem with vanishing/exploding gradient with >1 layers -> introduce better initialisations and other activation functions


    """
    def __init__(self,nb_layers,alpha=0.01,thr=1e-5,max_iter=1000):
                
                if nb_layers<=0:
                        raise ValueError("nb_layers only positive integer")
                self.alpha = alpha
                self.thr = thr
                self.max_iter=max_iter
                self.nb_layers=nb_layers
                


    def logit_cdf(self,x):
        return 1/(1+np.exp(-x))
    def logit_pdf(self,x):
        return np.exp(-x)/((1+np.exp(-x))**2)

    def initialise_coefs(self,X):
        a1 = np.random.normal(loc=0.0, scale=1, size=(1, 1))
        b0 = np.random.normal(loc=0.0, scale=1, size=(1, 1))
        b_first = np.random.normal(loc=0.0, scale=1, size=(X.shape[1], 1))
        c_first = np.random.normal(loc=0.0, scale=1, size=(1, 1))

        if self.nb_layers>1:
                b_multi_layers=np.random.normal(loc=0.0, scale=1, size=(1, self.nb_layers-1))
                c_multi_layers = np.random.normal(loc=0.0, scale=1, size=(1, self.nb_layers-1))
                return a1,b0,b_first,c_first,b_multi_layers,c_multi_layers
        
        return a1,b0,b_first,c_first,None,None


    def hidden_layer(self,index,X,b_first,c_first,b_multi_layers,c_multi_layers,derivative=False):
           
            i=0
            while i!=index:
                    
                    if i==0:
                            lin_comb=X@b_first+c_first
                            h_= self.logit_cdf(lin_comb)
                                   
                    else:
                            lin_comb=h_*b_multi_layers[:,i-1]+c_multi_layers[:,i-1]#it is a vector but if without : error as vector in position 0 
                            h_=self.logit_cdf(lin_comb) 
                    i=i+1
                            
            return h_ if not derivative  else self.logit_pdf(lin_comb)
    
    def probability(self,X,b_first,c_first,b_multi_layers,c_multi_layers,a1,b0):
            f_xi=a1*self.hidden_layer(self.nb_layers,X,b_first,c_first,b_multi_layers,c_multi_layers,derivative=False)+b0
            return self.logit_cdf(f_xi)
            
    def gradients(self,y,X,b_first,c_first,b_multi_layers,c_multi_layers,a1,b0):
            
            gradient_b_first_storage=np.zeros((X.shape[1], 1))#first b is in  Rd,  others in R
            gradient_b_multi_storage=np.zeros((1, self.nb_layers-1))
            gradient_c_all_storage=np.zeros((1, self.nb_layers))
            

            gradient_a1=self.hidden_layer(self.nb_layers,X,
                                                                           b_first,
                                                                           c_first,
                                                                           b_multi_layers,
                                                                           c_multi_layers,
                                                                           derivative=False
                                                                           )
            gradient_b0=1
            probability=self.probability(X,b_first,c_first,b_multi_layers,c_multi_layers,a1,b0)
            #parallelise
            for i in range(1,self.nb_layers+1):#from 1 to J layers
                    #vectors multiplication element by element 
                    product_phi_s_prime=np.multiply.reduce(np.array([self.hidden_layer(j,X,
                                                                           b_first,
                                                                           c_first,
                                                                           b_multi_layers,
                                                                           c_multi_layers,
                                                                           derivative=True
                                                                           ) for j in range(i,self.nb_layers+1) ]))
                   
                    result = np.array([b_multi_layers[:,j-1-i] for j in range(i+1, self.nb_layers+1)])
                    if result.size == 0:
                                result = np.array([1])
                    #scalars multiplication 
                    product_b_s=np.prod(result)

                    if i==1:
                         hidden_previous_layer=X
                    else:
                         hidden_previous_layer=self.hidden_layer(i-1,X,
                                                                           b_first,
                                                                           c_first,
                                                                           b_multi_layers,
                                                                           c_multi_layers,
                                                                           derivative=False
                                                                           )
                   

                    gradient_b=np.mean((y - probability) * a1 *product_phi_s_prime*product_b_s* hidden_previous_layer , axis=0)
                    gradient_c=np.mean((y - probability) * a1 *product_phi_s_prime*product_b_s,axis=0)
                    if i==1:
                            gradient_b=gradient_b.reshape(-1,1)
                            gradient_b_first_storage=gradient_b
                    else:
                        gradient_b_multi_storage[:,i-2]=gradient_b
                    gradient_c_all_storage[:,i-1]=gradient_c#.flatten()
            gradient_a1_storage=np.mean((y - probability) * gradient_a1, axis=0)
            gradient_b0_storage=np.mean((y - probability) * gradient_b0, axis=0)
            print("‖∇b_first‖ =", np.linalg.norm(gradient_b_first_storage))


       
            return gradient_b_first_storage,gradient_b_multi_storage,gradient_c_all_storage,gradient_a1_storage,gradient_b0_storage


    def gradient_update_ascent(self,y,X,b_first,c_first,b_multi_layers,c_multi_layers,a1,b0,alpha):
                    

                        (
                        gradient_b_first_storage,
                        gradient_b_multi_storage,
                        gradient_c_all_storage,
                        gradient_a1_storage,
                        gradient_b0_storage
                        ) = self.gradients(
                        y,
                        X,
                        b_first,
                        c_first,
                        b_multi_layers,
                        c_multi_layers,
                        a1,
                        b0
                        )


                        update_b0 =b0+alpha*gradient_b0_storage
    
                        update_a1=a1+alpha*gradient_a1_storage
    
                        update_b_first=b_first+alpha*gradient_b_first_storage

                        update_c_first=c_first+alpha*gradient_c_all_storage[:,0].reshape(-1,1)
                      

                        if self.nb_layers>1:
                                
                                update_c_multi=c_multi_layers+alpha*gradient_c_all_storage[:,1:]
                                update_b_multi=b_multi_layers+alpha*gradient_b_multi_storage
                                
                                return update_a1, update_b0, update_b_first, update_c_first,update_b_multi,update_c_multi
                        return update_a1, update_b0, update_b_first, update_c_first,None,None
    
    def adjust_vector_dim(self,c_multi,b_multi):
                for_teta_c_mult=None
                for_teta_b_mult=None
                if c_multi is not  None:

                  for_teta_c_mult=c_multi[0].reshape(-1,1)
                  for_teta_b_mult=b_multi[0].reshape(-1,1)
                return for_teta_c_mult,for_teta_b_mult
             

    def get_theta_no_None(self,*args):
            arrays = list(args)
            arrays_to_concat = [arr for arr in arrays if arr is not None]
            teta_old=np.concatenate(arrays_to_concat)
            return teta_old
    def gradient_ascent(self,X,y,alpha):
                
                a1_old,b0_old,b_first_old,c_first_old,b_multi_old,c_multi_old=self.initialise_coefs(X)


                for_teta_c_mult_old,for_teta_b_mult_old=self.adjust_vector_dim(c_multi_old,b_multi_old)

                
                teta_old=self.get_theta_no_None(a1_old,b0_old,b_first_old,c_first_old,for_teta_b_mult_old,for_teta_c_mult_old)
                dif=np.inf
                i=0
                while dif>self.thr:
                        i=i+1
                        
                        a1, b0, b_first, c_first,b_multi,c_multi=self.gradient_update_ascent(y,X,b_first_old,c_first_old,b_multi_old,c_multi_old,a1_old,b0_old,alpha)
                        
                        for_teta_c_mult,for_teta_b_mult=self.adjust_vector_dim(c_multi,b_multi)
                        
                        teta=self.get_theta_no_None(a1,b0,b_first,c_first,for_teta_b_mult,for_teta_c_mult)
                       
                        
                        dif=np.linalg.norm(teta - teta_old)
                        
                        teta_old=teta
                        a1_old,b0_old,b_first_old,c_first_old,b_multi_old,c_multi_old=a1,b0,b_first,c_first,b_multi,c_multi
                        if  i>self.max_iter:
                                print("reached max iterations, not converged")
                                break
                print(i)                                               
                self.a1 = a1
                self.b0 = b0
                self.b_first = b_first
                self.c_first = c_first
                self.b_multi = b_multi
                self.c_multi = c_multi

    def train(self,X,y):
                 y=y.reshape(-1,1)
                 self.gradient_ascent(X,y,self.alpha)


    def predict_proba(self,X):
                # probability(self,X,b_first,epsilon_first,b_multi_layers,epsilon_multi_layers,a1,b0):
                return self.probability(X,self.b_first,self.c_first,self.b_multi,self.c_multi,self.a1,self.b0)
    
    def predict(self,X,thr=0.5):
                predictions= (self.predict_proba(X)>thr).astype(int)
                return predictions.reshape(1,-1)
                
    

