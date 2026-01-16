# 🧠 Challenge : building a deep learning library in numpy


---

This repository accompanies the final project of the course ‘Advanced Machine Learning’ taught by Austin J. Stromme at ENSAE Paris (Institut Polytechnique de Paris).
It serves as a workspace for developing and experimenting with neural networks implemented purely in NumPy, including code development, small-scale experiments, theoretical derivations, and methodological explorations, for learning, research preparation, and interview readiness.

Reference document:
Theoretical report on which the implementation is based: [https://www.overleaf.com/read/pmbzgsrrdxws#08d781](https://www.overleaf.com/read/pmbzgsrrdxws#08d781)

---

Execution is available in main.ipynb
with installed required packages in requirements.txt


## 🛠️ Features 

### MLP

- Activation functions: Sigmoid, ReLU, Tanh
- Initializations: Random, Xavier, He, LeCun
- Optimizers: SGD, Mini-batch GD, Momentum, RMSProp, Adam
- Loss functions: Binary Cross-Entropy, Multi-class Cross-Entropy
- Regularization: L2, Dropout
- Training loop with early stopping
- Batch Normalization

### CNN

- NumPy-based convolutions and max-pooling with efficient implementations
- Custom asymmetric padding support
- Architecture foundation extendable towards AlexNet / VGG-style networks
- Trained prototype achieving ~90% accuracy on a CIFAR-10 subset with ~60k parameters




