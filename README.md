# 🧠 NN-from-Scratch

*A journey into building a deep learning library from the ground up.*

---

## 🚀 Motivation

Instead of importing `sklearn.neural_network.MLPClassifier` or TensorFlow’s `Dense` layer, this project is about **reinventing the wheel — on purpose**.  

The best way to truly understand:

- **Forward propagation**
- **Backpropagation**
- **Weight initialization**
- **Regularization**
- **Optimizers**
- **Batch_norm**

…is to code them by hand.

This repo is my sandbox for building and experimenting with a **Multi-Layer Perceptron (MLP)** from scratch. Other models TBD


The link to theoretical pdf: https://www.overleaf.com/read/pmbzgsrrdxws#08d781

---

## 🛠️ Features (so far)

- ✅ Custom `Layer` class to define architecture
- ✅ Strong parameter validation with friendly error messages
- ✅ Activation functions: **Sigmoid**, **ReLU**, **Tanh**
- ✅ Initializations: **Random**, **Xavier**, **He**, **LeCun**
- ✅ Additional optimizers: **SGD**, **Mini-batch GD**, **Adam**
- ✅ Loss functions: **Binary Cross-Entropy**, **Multi-class Cross-Entropy**
- ✅ Regularization: **L2**, **Dropout**
- ✅ Training loop with convergence check and early stopping.
- ✅ Batch-normalisation 


---

## 🔮 Roadmap (coming soon)

- ⏳ CNN


---

## 📖 Philosophy

This project isn’t about outperforming PyTorch or TensorFlow.  
It’s about **education**: exposing the internals of an MLP and exploring each design choice along the way.  

Think of it as building your own "mini-Keras" to really **feel the math** behind deep learning.

---

## ▶️ Quick Example

```python
import pandas as pd
from main import Layer, MLP_Classifier

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

