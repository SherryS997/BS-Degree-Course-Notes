---
title: Topics in Deep Learning
---

Unsupervised Pre-training
=====================================

The Challenge of Training Deep Neural Networks
---------------------------------------------

Training deep neural networks has been a long-standing challenge in machine learning. The main issue lies in the optimization of the network's weights, which is a complex and non-convex problem. Before 2006, training deep neural networks was difficult due to the vanishing gradient problem, where the gradients used to update the weights become smaller as they propagate through the network, making it hard to optimize the weights.

Unsupervised Pre-training: A Solution to the Problem
---------------------------------------------------

In 2006, a seminal work introduced the concept of unsupervised pre-training, which allows for the training of deep neural networks. The idea is to train one layer at a time, using an unsupervised objective function. This is done by reconstructing the input from the hidden layer, rather than predicting the output.

### Autoencoders

An autoencoder is a neural network that tries to reconstruct its input. It consists of an encoder that maps the input to a hidden representation, and a decoder that maps the hidden representation back to the input. The objective function is to minimize the difference between the input and the reconstructed input.

Let's denote the input vector as $\mathbf{x}$, the hidden representation as $\mathbf{h}$, and the reconstructed input as $\hat{\mathbf{x}}$. The autoencoder can be represented as:

$$
\mathbf{h} = \mathbf{g}^{(1)}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})
$$

$$
\hat{\mathbf{x}} = \mathbf{g}^{(2)}(\mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)})
$$

The objective function is to minimize the reconstruction error, which is typically measured using the mean squared error (MSE) or cross-entropy loss:

$$
\mathcal{L}(\theta) = \frac{1}{2} \left\lVert \mathbf{x} - \hat{\mathbf{x}} \right\rVert^2
$$

### Unsupervised Pre-training of Autoencoders

In unsupervised pre-training, the autoencoder is trained to reconstruct the input from the hidden layer. This is done by minimizing the reconstruction error using the objective function above. The hidden layer is trained to capture the important features of the input, and the decoder is trained to reconstruct the input from these features.

The unsupervised pre-training process is done in a greedy layer-wise manner. Each layer is trained independently, using the output of the previous layer as the input. This process is repeated until all layers are trained.

Let's denote the input to the $i$-th layer as $\mathbf{x}^{(i-1)}$, the hidden representation as $\mathbf{h}^{(i)}$, and the reconstructed input as $\hat{\mathbf{x}}^{(i-1)}$. The $i$-th layer can be represented as:

$$
\mathbf{h}^{(i)} = \mathbf{g}^{(i)}(\mathbf{W}^{(i)} \mathbf{x}^{(i-1)} + \mathbf{b}^{(i)})
$$

$$
\hat{\mathbf{x}}^{(i-1)} = \mathbf{g}^{(i+1)}(\mathbf{W}^{(i+1)} \mathbf{h}^{(i)} + \mathbf{b}^{(i+1)})
$$

The objective function for the $i$-th layer is to minimize the reconstruction error:

$$
\mathcal{L}(\theta^{(i)}) = \frac{1}{2} \left\lVert \mathbf{x}^{(i-1)} - \hat{\mathbf{x}}^{(i-1)} \right\rVert^2
$$

### Abstract Representation

Each layer in the network captures an abstract representation of the input. The first layer captures the most important features of the input, and each subsequent layer captures a more abstract representation of the previous layer.

### Supervised Fine-Tuning

Once all layers are trained using unsupervised pre-training, the network is fine-tuned using a supervised objective function. The weights learned during unsupervised pre-training are used as the initialization for the supervised fine-tuning.

Let's denote the output vector as $\mathbf{y}$, and the network's output as $\hat{\mathbf{y}}$. The supervised objective function is to minimize the difference between the output and the predicted output:

$$
\mathcal{L}(\theta) = \frac{1}{2} \left\lVert \mathbf{y} - \hat{\mathbf{y}} \right\rVert^2
$$

Why Unsupervised Pre-training Works
--------------------------------------

There are two possible reasons why unsupervised pre-training works: better optimization and better regularization.

### Better Optimization

Unsupervised pre-training can lead to better optimization of the network's weights. By training each layer independently, the network is able to converge to a better minimum of the loss function.

### Better Regularization

Unsupervised pre-training can also lead to better regularization of the network. By capturing the important features of the input, the network is able to generalize better to unseen data.

Impact of Unsupervised Pre-training
--------------------------------------

The work on unsupervised pre-training led to a series of advances in deep learning, including better optimization algorithms, regularization techniques, and initialization methods. It also sparked interest in designing better methods for optimization and regularization.

### Optimization Algorithms

One of the most popular optimization algorithms is Adam, which is an adaptive learning rate method. Adam adapts the learning rate for each parameter based on the magnitude of the gradient.

Let's denote the velocity at iteration $t$ as $\mathbf{u}_t$, the accumulated history of squared gradients at iteration $t$ as $v_t$, and the hyperparameter controlling the exponential decay rate as $\beta$. The update rule for Adam is:

$$
\mathbf{u}_t = \beta \mathbf{u}_{t-1} + (1 - \beta) \nabla \mathcal{L}_t
$$

$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla \mathcal{L}_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{\mathbf{u}_t}{\sqrt{v_t} + \epsilon}
$$

### Regularization Techniques

One of the most popular regularization techniques is dropout, which randomly drops out neurons during training. Dropout prevents the network from overfitting by making it difficult for the network to rely on any single neuron.

Let's denote the dropout rate as $p$, and the binary mask as $\mathbf{m}$. The output of the $i$-th layer with dropout is:

$$
\mathbf{h}^{(i)} = \mathbf{g}^{(i)}(\mathbf{W}^{(i)} \mathbf{x}^{(i-1)} + \mathbf{b}^{(i)}) \odot \mathbf{m}
$$

where $\odot$ denotes element-wise multiplication.

Optimization in Deep Neural Networks
=====================================

### The Optimization Problem

The optimization problem in deep neural networks involves minimizing the loss function, typically measured by the mean squared error (MSE), between the network's predictions and the true outputs. Given an input vector $\mathbf{x}$, the network's output is represented by $\hat{\mathbf{y}} = \mathbf{f}(\mathbf{x}; \theta)$, where $\theta$ denotes the model parameters. The loss function is defined as:

$$\mathcal{L}(\theta) = \frac{1}{2} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$

where $N$ is the total number of training samples, and $y_i$ is the true output for the $i$-th sample.

### Error Surface of Deep Neural Networks

The error surface of a deep neural network is highly non-convex, with many local minima, plateaus, and valleys. This makes it challenging to optimize the loss function using traditional optimization methods. The error surface can be visualized as a complex landscape with multiple minima, where the goal is to find the global minimum.

### Universal Approximation Theorem

The universal approximation theorem states that a sufficiently large neural network can approximate any continuous function to an arbitrary degree of precision. This implies that, in theory, it is possible to drive the training error to zero by increasing the capacity of the network. However, this comes at a cost, as larger networks require more time and resources to train.

### Capacity of Deep Neural Networks

The capacity of a deep neural network refers to its ability to fit the training data. Increasing the capacity of a neural network by adding more neurons or layers can lead to better optimization. However, this also increases the risk of overfitting, where the network becomes too specialized to the training data and fails to generalize well to new, unseen data.

### Pre-training and Optimization

Pre-training involves training a neural network on an unsupervised objective, such as reconstructing the input data, before fine-tuning it on the supervised objective. Some researchers argue that pre-training helps with optimization by allowing the network to find better local minima. However, others argue that pre-training is not necessary for optimization, and that a large enough neural network can achieve good optimization without pre-training.

Experiments have shown that pre-training can lead to better optimization, but this may be due to the increased capacity of the network rather than the pre-training itself. By increasing the capacity of the network, pre-training can provide a better initialization for the supervised objective, leading to faster convergence and better optimization.

### Regularization and Pre-training

Pre-training can also be seen as a form of regularization, as it constrains the weights to lie in certain regions of the parameter space. This is similar to L2 regularization, which constrains the weights to lie within a certain circle, and early stopping, which prevents the weights from growing too large. By constraining the weights, pre-training can lead to better generalization and optimization.

### Robustness to Random Initializations

Experiments have shown that pre-training can make deep neural networks more robust to random initializations. This means that the network is less sensitive to the initial weights and can achieve good optimization even with different initializations. This has led to research into better initialization methods, as well as better optimization and regularization methods.

### Importance of Initialization, Optimization, and Regularization

The research into pre-training has highlighted the importance of initialization, optimization, and regularization in deep neural networks. By focusing on these areas, researchers have been able to develop better methods for training deep neural networks, including better optimization algorithms, regularization methods, and weight initialization strategies.

Some of the popular optimization algorithms used in deep learning include:

* **Stochastic Gradient Descent (SGD)**: SGD is a simple and widely used optimization algorithm that updates the model parameters in the direction of the negative gradient of the loss function.

$$\theta_t = \theta_{t-1} - \eta \nabla \mathcal{L}_t$$

* **Momentum**: Momentum is an extension of SGD that incorporates a momentum term to help escape local minima.

$$\mathbf{u}_t = \beta \mathbf{u}_{t-1} + \eta \nabla \mathcal{L}_t$$

$$\theta_t = \theta_{t-1} - \mathbf{u}_t$$

* **Adagrad**: Adagrad is an optimization algorithm that adapts the learning rate for each parameter based on the gradient norm.

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla \mathcal{L}_t^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla \mathcal{L}_t$$

* **Adam**: Adam is a popular optimization algorithm that combines the benefits of momentum and Adagrad.

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla \mathcal{L}_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla \mathcal{L}_t^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t + \epsilon}} m_t$$

These optimization algorithms, along with regularization methods and weight initialization strategies, have been crucial in the development of deep learning models that can achieve state-of-the-art performance on a wide range of tasks.

Activation Functions
=====================

### Sigmoid Function

The sigmoid function, denoted by $\sigma(x)$, is a widely used activation function in neural networks. It is defined as:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

where $e$ is the base of the natural logarithm. The sigmoid function maps the input $x$ to a value between 0 and 1.

**Properties of Sigmoid Function**

* The sigmoid function is continuous and differentiable.
* The sigmoid function is monotonically increasing, meaning that as the input $x$ increases, the output $\sigma(x)$ also increases.
* The sigmoid function has an S-shaped curve, which allows it to model complex relationships between inputs and outputs.

**Derivative of Sigmoid Function**

The derivative of the sigmoid function is given by:

$$\sigma'(x) = \sigma(x) (1 - \sigma(x))$$

The derivative of the sigmoid function is used in backpropagation to compute the gradients of the loss function with respect to the model parameters.

**Limitations of Sigmoid Function**

* The sigmoid function has a limited range of outputs, which can lead to vanishing gradients during backpropagation.
* The sigmoid function is not zero-centered, which can cause the gradients to have different magnitudes for different inputs.
* The sigmoid function is computationally expensive, as it involves the computation of the exponential function.

### Tanh Function

The tanh function, denoted by $\tanh(x)$, is another widely used activation function in neural networks. It is defined as:

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

The tanh function maps the input $x$ to a value between -1 and 1.

**Properties of Tanh Function**

* The tanh function is continuous and differentiable.
* The tanh function is monotonically increasing, meaning that as the input $x$ increases, the output $\tanh(x)$ also increases.
* The tanh function has an S-shaped curve, which allows it to model complex relationships between inputs and outputs.

**Derivative of Tanh Function**

The derivative of the tanh function is given by:

$$\tanh'(x) = 1 - \tanh^2(x)$$

The derivative of the tanh function is used in backpropagation to compute the gradients of the loss function with respect to the model parameters.

**Limitations of Tanh Function**

* The tanh function has a limited range of outputs, which can lead to vanishing gradients during backpropagation.
* The tanh function is computationally expensive, as it involves the computation of the exponential function.

### Rectified Linear Unit (ReLU) Function

The ReLU function, denoted by $\text{ReLU}(x)$, is a widely used activation function in neural networks. It is defined as:

$$\text{ReLU}(x) = \max(0, x)$$

The ReLU function maps the input $x$ to a value between 0 and $x$.

**Properties of ReLU Function**

* The ReLU function is continuous but not differentiable at $x=0$.
* The ReLU function is monotonically increasing, meaning that as the input $x$ increases, the output $\text{ReLU}(x)$ also increases.
* The ReLU function is computationally efficient, as it only involves a simple thresholding operation.

**Derivative of ReLU Function**

The derivative of the ReLU function is given by:

$$\text{ReLU}'(x) = \begin{cases}
0 & \text{if } x < 0 \\
1 & \text{if } x \geq 0
\end{cases}$$

The derivative of the ReLU function is used in backpropagation to compute the gradients of the loss function with respect to the model parameters.

**Limitations of ReLU Function**

* The ReLU function can cause neurons to die, meaning that the output of the neuron becomes zero and remains zero during training.
* The ReLU function is not zero-centered, which can cause the gradients to have different magnitudes for different inputs.

### Leaky ReLU Function

The Leaky ReLU function, denoted by $\text{LeakyReLU}(x)$, is a variant of the ReLU function that allows a small fraction of the input to pass through even when the input is negative. It is defined as:

$$\text{LeakyReLU}(x) = \max(\alpha x, x)$$

where $\alpha$ is a small constant, typically set to 0.1.

**Properties of Leaky ReLU Function**

* The Leaky ReLU function is continuous and differentiable.
* The Leaky ReLU function is monotonically increasing, meaning that as the input $x$ increases, the output $\text{LeakyReLU}(x)$ also increases.
* The Leaky ReLU function is computationally efficient, as it only involves a simple thresholding operation.

**Derivative of Leaky ReLU Function**

The derivative of the Leaky ReLU function is given by:

$$\text{LeakyReLU}'(x) = \begin{cases}
\alpha & \text{if } x < 0 \\
1 & \text{if } x \geq 0
\end{cases}$$

The derivative of the Leaky ReLU function is used in backpropagation to compute the gradients of the loss function with respect to the model parameters.

**Parametric ReLU Function**

The Parametric ReLU function, denoted by $\text{PrELU}(x)$, is a variant of the ReLU function that allows the slope of the negative region to be learned during training. It is defined as:

$$\text{PrELU}(x) = \max(\alpha x, x)$$

where $\alpha$ is a learnable parameter.

**Properties of Parametric ReLU Function**

* The Parametric ReLU function is continuous and differentiable.
* The Parametric ReLU function is monotonically increasing, meaning that as the input $x$ increases, the output $\text{PrELU}(x)$ also increases.
* The Parametric ReLU function is computationally efficient, as it only involves a simple thresholding operation.

**Derivative of Parametric ReLU Function**

The derivative of the Parametric ReLU function is given by:

$$\text{PrELU}'(x) = \begin{cases}
\alpha & \text{if } x < 0 \\
1 & \text{if } x \geq 0
\end{cases}$$

The derivative of the Parametric ReLU function is used in backpropagation to compute the gradients of the loss function with respect to the model parameters.

### Exponential Linear Unit (ELU) Function

The ELU function, denoted by $\text{ELU}(x)$, is a variant of the ReLU function that allows the output to be negative. It is defined as:

$$\text{ELU}(x) = \begin{cases}
x & \text{if } x \geq 0 \\
\alpha (e^x - 1) & \text{if } x < 0
\end{cases}$$

where $\alpha$ is a small constant, typically set to 1.

**Properties of ELU Function**

* The ELU function is continuous and differentiable.
* The ELU function is monotonically increasing, meaning that as the input $x$ increases, the output $\text{ELU}(x)$ also increases.
* The ELU function is computationally efficient, as it only involves a simple thresholding operation.

**Derivative of ELU Function**

The derivative of the ELU function is given by:

$$\text{ELU}'(x) = \begin{cases}
1 & \text{if } x \geq 0 \\
\alpha e^x & \text{if } x < 0
\end{cases}$$

The derivative of the ELU function is used in backpropagation to compute the gradients of the loss function with respect to the model parameters.

### Softmax Activation Function

The Softmax activation function, denoted by $\sigma(x)$, is typically used as the output layer activation function in classification problems. It is defined as:

$$\sigma(x) = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}$$

where $x_i$ is the $i^{th}$ element of the input vector, and $K$ is the number of classes.

**Properties of Softmax Activation Function**

* The Softmax activation function is continuous and differentiable.
* The Softmax activation function is monotonically increasing, meaning that as the input increases, the output also increases.
* The Softmax activation function is a probability distribution, meaning that the output values sum up to 1.

**Derivative of Softmax Activation Function**

The derivative of the Softmax activation function is given by:

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

The derivative of the Softmax activation function is used in backpropagation to compute the gradients of the loss function with respect to the model parameters.

**Advantages of Softmax Activation Function**

* The Softmax activation function is suitable for classification problems, as it outputs a probability distribution over the classes.
* The Softmax activation function is differentiable, making it easy to optimize using gradient-based methods.
* The Softmax activation function is widely used and well-established in the deep learning community.

### Gelu Activation Function

The Gelu activation function, also known as the Gaussian Error Linear Unit, is a recently introduced activation function that has been shown to be more effective than ReLU and its variants. It is defined as:

$$f(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

**Properties of Gelu Activation Function**

* The Gelu activation function is continuous and differentiable.
* The Gelu activation function is non-monotonic, meaning that it can have multiple local maxima and minima.
* The Gelu activation function is self-gated, meaning that it can adaptively adjust the amount of input to pass through based on the input value.

**Derivative of Gelu Activation Function**

The derivative of the Gelu activation function is given by:

$$f'(x) = \Phi(x) + x \cdot \phi(x)$$

where $\phi(x)$ is the probability density function of the standard normal distribution.

**Advantages of Gelu Activation Function**

* The Gelu activation function has been shown to be more effective than ReLU and its variants in various deep learning tasks.
* The Gelu activation function is more robust to outliers and noisy data.
* The Gelu activation function can adaptively adjust the amount of input to pass through based on the input value.

### Swish Activation Function

The Swish activation function is a recently introduced activation function that is a generalization of the Gelu activation function. It is defined as:

$$f(x) = x \cdot \sigma(\beta x)$$

where $\sigma(x)$ is the sigmoid function and $\beta$ is a learnable parameter.

**Properties of Swish Activation Function**

* The Swish activation function is continuous and differentiable.
* The Swish activation function is non-monotonic, meaning that it can have multiple local maxima and minima.
* The Swish activation function is self-gated, meaning that it can adaptively adjust the amount of input to pass through based on the input value.

**Derivative of Swish Activation Function**

The derivative of the Swish activation function is given by:

$$f'(x) = \sigma(\beta x) + x \cdot \beta \cdot \sigma'(\beta x)$$

where $\sigma'(x)$ is the derivative of the sigmoid function.

**Advantages of Swish Activation Function**

* The Swish activation function has been shown to be more effective than ReLU and its variants in various deep learning tasks.
* The Swish activation function is more robust to outliers and noisy data.
* The Swish activation function can adaptively adjust the amount of input to pass through based on the input value.

### Silu Activation Function

The Silu activation function, also known as the Sigmoid Weighted Linear Unit, is a recently introduced activation function that is a specialization of the Swish activation function. It is defined as:

$$f(x) = x \cdot \sigma(x)$$

where $\sigma(x)$ is the sigmoid function.

**Properties of Silu Activation Function**

* The Silu activation function is continuous and differentiable.
* The Silu activation function is non-monotonic, meaning that it can have multiple local maxima and minima.
* The Silu activation function is self-gated, meaning that it can adaptively adjust the amount of input to pass through based on the input value.

**Derivative of Silu Activation Function**

The derivative of the Silu activation function is given by:

$$f'(x) = \sigma(x) + x \cdot \sigma'(x)$$

where $\sigma'(x)$ is the derivative of the sigmoid function.

**Advantages of Silu Activation Function**

* The Silu activation function has been shown to be more effective than ReLU and its variants in various deep learning tasks.
* The Silu activation function is more robust to outliers and noisy data.
* The Silu activation function can adaptively adjust the amount of input to pass through based on the input value.

