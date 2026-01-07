# 🧠 Challenge : building a deep learning library in numpy


---

This repository accompanies the final project of the course ‘Advanced Machine Learning’ taught by Austin J. Stromme at ENSAE Paris (Institut Polytechnique de Paris).
It serves as a workspace for developing and experimenting with neural networks implemented purely in NumPy, including code development, small-scale experiments, theoretical derivations, and methodological explorations, for learning, research preparation, and interview readiness.


The link to theoretical pdf on which code was inspired: [https://www.overleaf.com/read/pmbzgsrrdxws#08d781](https://www.overleaf.com/read/pmbzgsrrdxws#08d781)

---

## 🛠️ Features (so far)

### MLP


- ✅ Activation functions: **Sigmoid**, **ReLU**, **Tanh**
- ✅ Initializations: **Random**, **Xavier**, **He**, **LeCun**
- ✅ Additional optimizers: **SGD**, **Mini-batch GD**, **Adam**, **Rmsprop**, **Momentum**
- ✅ Loss functions: **Binary Cross-Entropy**, **Multi-class Cross-Entropy**
- ✅ Regularization: **L2**, **Dropout**
- ✅ Training loop with early stopping
- ✅ Batch-normalisation 

### CNN

- ✅ numpy convolutions, maxpoolings and their gradients (faster then for loops)
- ✅ custom assymetric padding
- ✅ full generalized infrastructure that can be extended in more recent models such as Alexnet and VGG
- ✅ trained small model with 60K parameters only, achieving 90% of accuracy on sample of CIFAR10
---



## ▶️ Quick Example on MLP 

```python
import pandas as pd
from mlp import Layer, MLP_Classifier

# Define pandas dataset
#....
# Define architecture
layers = (
    Layer(8, "tanh", regul=("dropout", 0.5), initial="xavier",batchnorm=True),
    Layer(4, "relu", regul=("l2", 0.9), initial="he",),
    Layer(100, "relu", initial="he",law="uniform"),
)

# Instantiate model
mlp = MLP_Classifier(layers, alpha=0.01, max_iter=1000, seed=42, optim="adam", batch_size=100, nb_epochs_early_stopping=50)

# Train (X, Y must be pandas DataFrames)
mlp.train(X_train, Y_train)

#or with test set just to see validation data set performance during training
mlp.train(X_train, Y_train,X_test,Y_test)
# Predict
preds = mlp.predict(X_test)
```

## ▶️ Quick Example on CNN

q=CNN(
    (
        ConvLayer(in_channels=3,output_channels=16,kernel_size=3,stride=1),
        MaxPoolLayer(kernel_size=3,stride=2),
        ConvLayer(in_channels=16,output_channels=32,kernel_size=3,stride=1),
        MaxPoolLayer(kernel_size=3,stride=2),
        FlatLayer(),
        Layer(
                nb_neurons=64,
                activation_function="relu",
                regul=("l2", 0.001),
                initial="he",
                batchnorm=True
            ),
            Layer(
                nb_neurons=32,
                activation_function="relu",
                regul=("l2", 0.001),
                initial="he",  
            ),
    ),
    max_iter=30,
    thr=1e-5,
    alpha=0.05,
    seed=123,
    batch_size=500,
)

q.train(
     
        X_train,
        Y_train,
        X_test ,
        Y_test ,
        fct  = accuracy_score
)

