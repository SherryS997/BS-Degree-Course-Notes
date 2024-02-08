# Feed Forward Neural Networks and Back Propagation

Feed forward neural networks are a fundamental architecture in the realm of artificial neural networks (ANNs), designed to process data in a forward direction, from input to output. This architecture is crucial for various machine learning tasks, including classification, regression, and pattern recognition. In this section, we delve into the intricacies of feed forward neural networks, including their components, computation processes, parameter learning techniques, and the crucial back propagation algorithm.

## Components of a Feed Forward Neural Network

### Input Layer
The input layer serves as the entry point for data into the neural network. It consists of an $n$-dimensional vector, where each element represents a feature or attribute of the input data. Mathematically, the input layer can be represented as:

$$
\mathbf{x} \in \mathbb{R}^n
$$

Here, $\mathbf{x}$ denotes the input vector, and $n$ represents the number of input features.

### Hidden Layers
Hidden layers form the core computational units of a feed forward neural network. These layers, typically denoted as $L - 1$, are responsible for processing and transforming the input data through a series of non-linear transformations. Each hidden layer comprises a set of neurons, with each neuron connected to every neuron in the previous layer. Mathematically, the $i$-th hidden layer can be represented as:

$$
\text{Layer } i : \mathbf{a}^{(i)} = \mathbf{g}^{(i)}(\mathbf{z}^{(i)})
$$

Here, $\mathbf{a}^{(i)}$ represents the activation vector of the $i$-th layer, $\mathbf{z}^{(i)}$ denotes the pre-activation vector, and $\mathbf{g}^{(i)}$ represents the activation function applied element-wise to $\mathbf{z}^{(i)}$.

### Output Layer
The output layer is the final layer of the neural network, responsible for generating the network's predictions or outputs. The number of neurons in the output layer depends on the nature of the task (e.g., binary classification, multi-class classification, regression). Mathematically, the output layer can be represented as:

$$
\text{Output Layer: } \mathbf{y} = \mathbf{f}(\mathbf{a}^{(L)})
$$

Here, $\mathbf{y}$ represents the output vector, and $\mathbf{f}$ denotes the output activation function.

## Computing Pre-activation and Activation

The computation process in a feed forward neural network involves computing the pre-activation and activation values for each neuron in the network.

### Pre-activation
Pre-activation refers to the linear transformation applied to the input data, followed by the addition of a bias term. Mathematically, the pre-activation for the $i$-th layer can be expressed as:

$$
\mathbf{z}^{(i)} = \mathbf{W}^{(i)} \mathbf{a}^{(i-1)} + \mathbf{b}^{(i)}
$$

Here, $\mathbf{W}^{(i)}$ represents the weight matrix connecting the $(i-1)$-th and $i$-th layers, $\mathbf{a}^{(i-1)}$ denotes the activation vector of the previous layer, and $\mathbf{b}^{(i)}$ represents the bias vector for the $i$-th layer.

### Activation
Activation involves applying a non-linear function to the pre-activation values, introducing non-linearity into the network's computations. Common activation functions include sigmoid, tanh, ReLU, and softmax. Mathematically, the activation for the $i$-th layer can be expressed as:

$$
\mathbf{a}^{(i)} = \mathbf{g}^{(i)}(\mathbf{z}^{(i)})
$$

Where $\mathbf{g}^{(i)}$ represents the activation function applied element-wise to $\mathbf{z}^{(i)}$.

## Output Activation and Function Approximation

The output activation function plays a crucial role in determining the nature of the network's predictions. Depending on the task at hand, different activation functions may be employed to ensure appropriate output scaling and behavior.

### Output Activation Function
The output activation function governs the transformation of the final layer's pre-activation values into the network's outputs. Common choices include softmax for multi-class classification tasks and linear functions for regression tasks. Mathematically, the output activation function can be expressed as:

$$
\mathbf{y} = \mathbf{f}(\mathbf{a}^{(L)})
$$

Where $\mathbf{f}$ denotes the output activation function.

### Function Approximation
The feed forward neural network serves as a powerful function approximator, capable of capturing complex relationships between inputs and outputs. By iteratively adjusting the network's parameters through training, the network learns to approximate the underlying function mapping inputs to outputs. Mathematically, the network's output ($\hat{\mathbf{y}}$) can be expressed as:

$$
\hat{\mathbf{y}} = \mathbf{f}(\mathbf{x}; \mathbf{W}, \mathbf{b})
$$

Where $\mathbf{W}$ and $\mathbf{b}$ represent the network's parameters, and $\mathbf{x}$ denotes the input vector.

## Parameter Learning and Loss Function

The process of learning in a feed forward neural network involves optimizing the network's parameters to minimize a predefined loss function. This optimization process typically utilizes gradient-based techniques, such as gradient descent, coupled with the back propagation algorithm.

### Parameter Optimization
The parameters of a feed forward neural network, including weights ($\mathbf{W}$) and biases ($\mathbf{b}$), are learned through iterative optimization algorithms. The objective is to minimize the discrepancy between the network's predictions and the true target values.

### Loss Function
The loss function quantifies the disparity between the predicted outputs of the network and the actual target values. Common choices for the loss function include the squared error loss for regression tasks and the categorical cross-entropy loss for classification tasks.

## Back Propagation Algorithm

The back propagation algorithm serves as the cornerstone of parameter learning in feed forward neural networks. It facilitates the efficient computation of gradients with respect to network parameters, enabling gradient-based optimization techniques to adjust the parameters iteratively.

### Forward Pass
During the forward pass, input data is propagated through the network, and pre-activation and activation values are computed for each layer.

### Backward Pass
During the backward pass,

 gradients of the loss function with respect to network parameters are computed recursively using the chain rule of calculus. These gradients are then used to update the parameters in the direction that minimizes the loss function.

### Update Rule
The update rule dictates how the network parameters are adjusted based on the computed gradients. Common choices include gradient descent, stochastic gradient descent, and variants such as Adam and RMSprop.


# Intuition in Feedforward Neural Networks

In this section, we delve into the intricacies of learning parameters for feedforward neural networks, elucidating the underlying principles and algorithms involved. We begin by revisiting the fundamental concepts of gradient descent and then extend our discussion to encompass the complexities introduced by the architecture of feedforward neural networks. Through meticulous examination, we elucidate the process of parameter learning, addressing key questions regarding the choice of loss function and efficient computation of partial derivatives.

## Gradient Descent Revisited

Gradient descent serves as a cornerstone algorithm in the realm of neural network training, facilitating the iterative adjustment of parameters to minimize the loss function. Mathematically, the process can be succinctly represented as follows:

$$
\theta_{t+1} = \theta_{t} - \alpha \cdot \nabla_{\theta} \mathcal{L}(\theta_t)
$$

Here, $\theta$ symbolizes the parameters of the neural network, $\alpha$ denotes the learning rate, and $\nabla_{\theta} \mathcal{L}(\theta_t)$ signifies the gradient of the loss function with respect to the parameters at iteration $t$.

## Transition to Feedforward Neural Networks

Moving beyond the realm of single neurons, we extend our focus to encompass feedforward neural networks, characterized by their layered architecture and interconnected nodes. In this context, the parameters of interest include weight matrices $\mathbf{W}^{(i)}$ and bias vectors $\mathbf{b}^{(i)}$ for each layer $i$.

## Parameter Representation

In contrast to the simplistic parameter representation in single neurons, where $\theta$ encapsulated only a handful of parameters, the scope expands significantly in feedforward neural networks. Now, $\theta$ encompasses a multitude of elements, incorporating the weights and biases across all layers of the network. Mathematically, we express this as:

$$
\theta = (\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \ldots, \mathbf{W}^{(L)}, \mathbf{b}^{(L)})
$$

Here, $L$ denotes the total number of layers in the network.

## Complexity of Parameters

With the proliferation of layers and neurons in feedforward neural networks, the parameter space expands exponentially, posing computational challenges. Despite this complexity, the fundamental principles of gradient descent remain applicable, albeit with adaptations to accommodate the increased dimensionality of the parameter space.

## Algorithm Adaptation

The essence of gradient descent persists in the context of feedforward neural networks, albeit with modifications to accommodate the augmented parameter space. The core objective remains unchanged: iteratively updating parameters to minimize the loss function. Through meticulous computation of gradients, facilitated by techniques such as backpropagation, the network adjusts its parameters to optimize performance.

## Challenges in Parameter Learning

The transition to feedforward neural networks introduces several challenges in the realm of parameter learning. Chief among these challenges is the computation of gradients, which necessitates the derivation of partial derivatives with respect to each parameter. In the context of complex architectures, this process can be computationally intensive and prone to errors.

## Choice of Loss Function

Central to the parameter learning process is the selection of an appropriate loss function, which quantifies the disparity between predicted and actual outputs. The choice of loss function is contingent upon the nature of the task at hand, with options ranging from mean squared error for regression tasks to cross-entropy loss for classification problems.

## Efficient Computation of Gradients

Efficient computation of gradients is paramount in the realm of parameter learning, particularly in the context of feedforward neural networks with intricate architectures. Techniques such as vectorization and parallelization play a pivotal role in enhancing computational efficiency, enabling rapid convergence during training.

# Output and Loss Functions

## Regression Problems
Regression problems involve predicting continuous values based on input data. For instance, in predicting movie ratings, the goal is to estimate a numerical value (rating) for each input (movie). 

### Loss Function: Mean Squared Error (MSE)
The mean squared error (MSE) is a common choice for regression tasks. It quantifies the average squared difference between the predicted and true values. Mathematically, MSE is expressed as:

$$
\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

where:

- $N$ is the number of training examples,
- $\hat{y}_i$ is the predicted value for the $i$-th example,
- $y_i$ is the true value for the $i$-th example.

### Output Function: Linear Activation
In regression tasks, a linear activation function is often employed at the output layer. This choice allows the model to produce unbounded output values, accommodating the natural range of the target variable. The output $\hat{\mathbf{y}}$ is computed as a linear transformation of the last hidden layer activations:

$$
\hat{\mathbf{y}} = \mathbf{W}^{(L)} \mathbf{a}^{(L-1)} + \mathbf{b}^{(L)}
$$

where $\mathbf{W}^{(L)}$ and $\mathbf{b}^{(L)}$ are the weight matrix and bias vector of the output layer, respectively.

## Classification Problems
Classification tasks involve assigning input data to discrete categories or classes. For example, in image classification, the aim is to categorize images into predefined classes.

### Output Function: Softmax Activation
To obtain probabilities for each class in a classification problem, the softmax activation function is commonly used at the output layer. Softmax transforms the raw scores (logits) into a probability distribution over the classes. The softmax function is defined as:

$$
\text{Softmax}(\mathbf{z}^{(L)})_i = \frac{e^{z_i^{(L)}}}{\sum_{j=1}^{K} e^{z_j^{(L)}}}, \quad i = 1, 2, \ldots, K
$$

where:

- $K$ is the number of classes,
- $\mathbf{z}^{(L)}$ is the pre-activation vector at the output layer.

### Loss Function: Cross Entropy
Cross entropy is a commonly used loss function for classification tasks. It measures the dissimilarity between the predicted probability distribution and the true distribution of class labels. The cross entropy loss is given by:

$$
\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})
$$

where:

- $N$ is the number of training examples,
- $K$ is the number of classes,
- $y_{i,k}$ is the indicator function for the $k$-th class of the $i$-th example,
- $\hat{y}_{i,k}$ is the predicted probability of the $k$-th class for the $i$-th example.

The cross entropy loss penalizes deviations between the predicted and true class probabilities, encouraging the model to assign high probabilities to the correct classes.

## Softmax Function
The softmax function is employed to convert raw scores into probabilities for multiclass classification tasks. It ensures that the output represents a valid probability distribution over the classes, with values between 0 and 1 that sum up to 1. The softmax function is mathematically defined as:

$$
\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i = 1, 2, \ldots, K
$$

where $\mathbf{z}$ is the input vector, and $K$ is the number of classes.

## Cross Entropy Loss
Cross entropy loss quantifies the difference between the predicted and true distributions of class labels in classification tasks. It is a fundamental component in training neural networks for classification. The cross entropy loss is given by the formula:

$$
\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})
$$

where:

- $N$ is the number of training examples,
- $K$ is the number of classes,
- $y_{i,k}$ is the indicator function for the $k$-th class of the $i$-th example,
- $\hat{y}_{i,k}$ is the predicted probability of the $k$-th class for the $i$-th example.

The cross entropy loss penalizes deviations between the predicted and true class probabilities. Minimizing this loss encourages the model to produce accurate probability distributions over the classes.

# Understanding Back Propagation Algorithm

## Derivatives and Chain Rule

### Derivative Calculation
In the context of neural networks, the derivative of the loss function with respect to the parameters (weights and biases) is essential for updating these parameters during the training process. This derivative quantifies how changes in the parameters affect the overall loss.

### Challenges in Deep Neural Networks
Unlike simpler networks, deep neural networks entail a more complex structure with multiple layers and numerous parameters. Computing derivatives in such networks requires careful consideration and efficient algorithms.

### Leveraging the Chain Rule
The chain rule of calculus provides a systematic approach to compute derivatives in composite functions. In the context of neural networks, it enables the computation of derivatives layer by layer, propagating the error from the output layer to the input layer.

## Chain Rule Intuition

### Step-by-Step Derivative Calculation
Visualizing the computation of derivatives as a chain of functions helps in understanding the iterative nature of back propagation. Each layer in the network contributes to the overall derivative calculation, with the chain rule facilitating this process.

### Reusability of Computations
Once a segment of the derivative chain is computed, it can be reused for similar computations across different parameters. This reusability reduces redundancy and computational complexity, making the back propagation algorithm more efficient.

### Generalization Across Layers
The principles of back propagation can be generalized across different layers and parameters in the network. By establishing a unified framework for derivative computation, the algorithm becomes more scalable and adaptable to varying network architectures.

## Responsibilities in Back Propagation

### Error Propagation
Back propagation involves tracing the propagation of errors from the output layer back to the input layer through the network's connections. Each layer in the network bears responsibility for contributing to this error propagation process.

### Influence of Weights and Biases
The weights and biases in the network play a crucial role in determining the magnitude of error propagation. Adjusting these parameters based on their influence on the loss function is key to optimizing the network's performance.

### Derivatives as Indicators of Influence
The derivatives of the loss function with respect to the parameters serve as indicators of their influence on the overall loss. Larger derivatives imply stronger influence, guiding the optimization process towards more effective parameter adjustments.

## Mathematical Realization

### Derivatives and Responsibilities
Mathematically, derivatives quantify the sensitivity of the loss function to changes in the parameters. By computing these derivatives, the algorithm assigns responsibilities to each parameter based on its impact on the overall loss.

### Partial Derivatives
Partial derivatives measure how the loss function changes with infinitesimal adjustments to individual parameters. This information guides the gradient-based optimization process, enabling efficient parameter updates.

### Objective of Back Propagation
The primary objective of back propagation is to compute gradients with respect to various components of the network, including output, hidden units, weights, and biases. These gradients drive the optimization process towards minimizing the loss function.

### Emphasis on Cross Entropy
In classification problems, where the network's output is represented using softmax activation, cross-entropy loss is commonly used. Back propagation algorithms are tailored to handle such loss functions efficiently, facilitating effective training of classification models.

# Gradient w.r.t output units

## Talking to the Output Layer

### Goal
The primary objective in back propagation is to compute the derivative of the loss function with respect to the output layer activations. Let's denote the output vector as $\mathbf{y}$, representing the network's predictions or outputs.

### Loss Function
The loss function, denoted as $\mathcal{L}(\theta_t)$, measures the discrepancy between the predicted output $\hat{\mathbf{y}}$ and the true labels $\mathbf{y}$. It is often defined as the negative logarithm of the predicted probability of the true class.

$$
\mathcal{L}(\theta_t) = -\log(\hat{y}_l)
$$

where $l$ is the true class label.

### Derivative Calculation
We aim to compute the derivative of the loss function with respect to each output neuron activation. This involves determining how a change in each output activation affects the overall loss. 

The derivative can be expressed as follows:

$$
\frac{\partial \mathcal{L}(\theta_t)}{\partial a^{(L)}_i} = 
\begin{cases}
-\frac{1}{\hat{y}_l} & \text{if } i = l \\
0 & \text{otherwise}
\end{cases}
$$

where $a^{(L)}_i$ represents the $i$-th output neuron activation, and $\hat{y}_l$ is the predicted probability corresponding to the true class label.

### Gradient Vector
The gradient of the loss function with respect to the output layer, denoted as $\nabla_{\mathbf{y}} \mathcal{L}(\theta_t)$, is a vector containing the partial derivatives of the loss function with respect to each output neuron activation. It can be represented as:

$$
\nabla_{\mathbf{y}} \mathcal{L}(\theta_t) = \begin{bmatrix}
-\frac{1}{\hat{y}_1} & 0 & \cdots & 0 \\
0 & -\frac{1}{\hat{y}_2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & -\frac{1}{\hat{y}_k}
\end{bmatrix}
$$

This gradient vector provides insights into how changes in the output layer activations affect the loss function.

## Talking to the Hidden Layers

### Objective
After understanding the derivatives at the output layer, the next step is to compute the derivatives with respect to the pre-activation values of the hidden layers. This involves understanding how changes in the pre-activations affect the output activations and, consequently, the loss function.

### Chain Rule Application
To compute the derivative of the loss function with respect to the pre-activation values of the hidden layers, we apply the chain rule. This breaks down the computation into two steps:

1. Derivative of the loss function with respect to the output activations.
2. Derivative of the output activations with respect to the pre-activation values.

### Pre-Activation to Activation
The pre-activation values of the hidden layers are passed through an activation function to obtain the output activations. Mathematically, this can be expressed as:

$$
\mathbf{a}^{(i)} = \mathbf{g}^{(i)}(\mathbf{z}^{(i)})
$$

where $\mathbf{z}^{(i)}$ represents the pre-activation vector for the $i$-th layer, and $\mathbf{g}^{(i)}(\cdot)$ is the activation function applied element-wise.

### Derivative Calculation
The derivative of the output activations with respect to the pre-activation values depends on the choice of activation function. For commonly used activation functions like sigmoid, tanh, and ReLU, the derivatives can be computed analytically.

### Gradient Flow
Understanding the gradient flow from the output layer to the hidden layers is crucial for parameter updates during training. The gradients propagate backward through the network, allowing for efficient computation of parameter updates.

## Talking to the Weights

### Objective
Once the derivatives with respect to the pre-activation values are computed, the next step is to calculate the derivatives with respect to the weights connecting the neurons. This step enables us to understand how changes in the weights influence the loss function.

### Chain Rule Application
Similar to the computation at the hidden layers, we apply the chain rule to compute the derivatives of the loss function with respect to the weights. This involves breaking down the computation into two parts:

1. Derivative of the loss function with respect to the output activations.
2. Derivative of the output activations with respect to the pre-activation values.

### Derivative Calculation
The derivative of the pre-activation values with respect to the weights connecting the neurons can be straightforwardly calculated using the input vector, output activations, and the derivative of the activation function.

### Weight Update
Once the derivatives with respect to the weights are computed, they are used to update the weights through optimization algorithms like gradient descent. By iteratively updating the weights based on the computed gradients, the network learns to minimize the loss function and improve its performance on the given task.

# Computing Gradients with Respect to Hidden Units

## Introduction to Hidden Units

Hidden units, also known as hidden layers, are intermediary layers in neural networks responsible for capturing complex patterns in the input data. These layers play a crucial role in the network's ability to learn and generalize from the training data.

## Chain Rule for Gradient Computation

The chain rule of calculus is a fundamental concept used extensively in computing derivatives of composite functions. In the context of neural networks, where the activation of each layer depends on the activations of the previous layers, the chain rule becomes essential for gradient computation.

Mathematically, let $P(Z)$ be a function dependent on intermediate functions $Q_1(Z), Q_2(Z),$ etc., and $P$ being a function of $Z$. The derivative of $P$ with respect to $Z$ is computed as follows:

$$
\frac{dP}{dZ} = \sum_{i=1}^{m} \frac{dP}{dQ_i} \cdot \frac{dQ_i}{dZ}
$$

Here, we sum over all paths from $Z$ to $P$, multiplying the derivatives along each path.

## Deriving the Formula

To compute gradients for hidden units, we apply the chain rule to derive a generic formula. Consider a specific hidden unit $H_{ij}$, where $i$ denotes the layer number and $j$ represents the neuron number within that layer.

We aim to compute the derivative of the loss function with respect to $H_{ij}$. This involves summing over all paths from $H_{ij}$ to the loss function, considering each path's contribution via the chain rule.

## Computing Gradients for Hidden Units

The derivative of the loss function with respect to $H_{ij}$ can be expressed as a dot product between two vectors:

$$
\frac{d\mathcal{L}}{dH_{ij}} = \mathbf{W}^{(i+1)}_j \cdot \frac{d\mathcal{L}}{d\mathbf{a}^{(i+1)}}
$$

Here, $\mathbf{W}^{(i+1)}_j$ represents the $j$-th column of the weight matrix connecting the $(i+1)$-th and $i$-th layers, and $\frac{d\mathcal{L}}{d\mathbf{a}^{(i+1)}}$ denotes the gradient of the loss function with respect to the activations in the next layer.

This computation involves the element-wise multiplication of the weight vector and the gradient vector.

## Generalizing the Formula

We generalize the formula to compute gradients for any hidden layer $H_i$ with multiple units. The derivative of the loss function with respect to $H_i$ is given by:

$$
\frac{d\mathcal{L}}{d\mathbf{H}_i} = \mathbf{W}^{(i+1)T} \cdot \frac{d\mathcal{L}}{d\mathbf{a}^{(i+1)}}
$$

Here, $\mathbf{W}^{(i+1)T}$ denotes the transpose of the weight matrix connecting the $(i+1)$-th and $i$-th layers, and $\frac{d\mathcal{L}}{d\mathbf{a}^{(i+1)}}$ represents the gradient of the loss function with respect to the activations in the next layer.

This formulation enables efficient computation of gradients for hidden units across all layers of the neural network.

# Derivatives with Respect to Parameters

## Computing Derivatives of Loss Function

### Iterative Approach

Rather than computing the derivatives of the loss function with respect to all parameters simultaneously, we adopt an iterative approach. This involves focusing on one parameter at a time, specifically one element of the weight matrix or one element of the bias vector.

### Derivative with Respect to Weight Matrix Element

Consider the derivative of the loss function with respect to one element of the weight matrix, $w_{ij}^{(k)}$, connecting the $(k-1)$-th and $k$-th layers. This derivative is obtained iteratively.

#### Derivative with Respect to Activation

First, compute the derivative of the loss function with respect to the corresponding activation, $a_{i}^{(k)}$, using chain rule.

#### Derivative of Activation with Respect to Weight

Next, compute the derivative of the activation with respect to $w_{ij}^{(k)}$, denoted as $\frac{\partial a_{i}^{(k)}}{\partial w_{ij}^{(k)}}$.

##### Mathematical Formulation

Mathematically, this derivative equals the activation of the preceding layer at index $i$, denoted as $h_{ij}^{(k-1)}$.

$$\frac{\partial a_{i}^{(k)}}{\partial w_{ij}^{(k)}} = h_{ij}^{(k-1)}$$

### Outer Product Representation

The derivative of the loss function with respect to $w_{ij}^{(k)}$ can be expressed as the outer product of two vectors: the derivative of the loss function with respect to the activations ($\mathbf{a}^{(k)}$) and the activations of the preceding layer ($\mathbf{h}^{(k-1)}$).

#### Mathematical Representation

$$\frac{\partial \mathcal{L}(\theta_t)}{\partial w_{ij}^{(k)}} = \frac{\partial \mathcal{L}(\theta_t)}{\partial \mathbf{a}^{(k)}} \otimes \mathbf{h}^{(k-1)}$$

where $\otimes$ represents the outer product operation.

### Efficient Computation

Both the quantities involved in the derivative computation can be efficiently computed during the forward pass of the neural network, requiring no additional computations during the backward pass.

## Derivative with Respect to Bias

Similar to the approach for weight matrices, the derivative of the loss function with respect to the bias vector ($\mathbf{b}^{(k)}$) is computed iteratively.

### Splitting into Two Parts

#### Derivative with Respect to Activation

First, compute the derivative of the loss function with respect to the activations ($\mathbf{a}^{(k)}$) using chain rule.

#### Derivative of Activation with Respect to Bias

Next, compute the derivative of the activation with respect to the bias vector, denoted as $\frac{\partial a_{i}^{(k)}}{\partial b_{i}^{(k)}}$.

##### Mathematical Formulation

Mathematically, this derivative is simply 1, as the bias term directly contributes to the activation.

$$\frac{\partial a_{i}^{(k)}}{\partial b_{i}^{(k)}} = 1$$

### Gradient Vector

The derivative of the loss function with respect to the bias vector ($\mathbf{b}^{(k)}$) is obtained by collecting all the partial derivatives, representing the gradient of the loss function with respect to the activations.

#### Mathematical Representation

$$\frac{\partial \mathcal{L}(\theta_t)}{\partial \mathbf{b}^{(k)}} = \frac{\partial \mathcal{L}(\theta_t)}{\partial \mathbf{a}^{(k)}}$$

# Backpropagation Algorithm

## Forward Propagation

### Overview

Forward propagation refers to the process of computing the network's output given an input. It involves passing the input data through the network layers, computing pre-activations and activations, and finally obtaining the network's predictions.

### Computation of Pre-activations and Activations

For each layer $i$ in the network, forward propagation involves the following steps:

1. **Pre-activation**: Compute the pre-activation vector $\mathbf{z}^{(i)}$ using the formula:
   $$\mathbf{z}^{(i)} = \mathbf{W}^{(i)} \mathbf{a}^{(i-1)} + \mathbf{b}^{(i)}$$
   Here, $\mathbf{W}^{(i)}$ is the weight matrix connecting the $(i-1)$-th and $i$-th layers, $\mathbf{a}^{(i-1)}$ is the activation vector from the previous layer, and $\mathbf{b}^{(i)}$ is the bias vector for the $i$-th layer.

2. **Activation**: Apply the activation function $\mathbf{g}^{(i)}(\cdot)$ element-wise to the pre-activation vector $\mathbf{z}^{(i)}$ to obtain the activation vector $\mathbf{a}^{(i)}$:
   $$\mathbf{a}^{(i)} = \mathbf{g}^{(i)}(\mathbf{z}^{(i)})$$

3. **Output Activation**: For the output layer, apply a specific output activation function $\mathbf{f}(\cdot)$ to obtain the final output $\hat{\mathbf{y}}$:
   $$\hat{\mathbf{y}} = \mathbf{f}(\mathbf{z}^{(L)})$$

## Loss Computation

After forward propagation, the next step is to compute the loss function, which measures the difference between the predicted output $\hat{\mathbf{y}}$ and the true output $\mathbf{y}$.

### Loss Function

The loss function $\mathcal{L}(\theta_t)$ is a measure of the error between the predicted and true outputs. It depends on the specific task and can be chosen based on the problem domain. Common loss functions include mean squared error (MSE), cross-entropy loss, and hinge loss.

### Loss Computation

Given the predicted output $\hat{\mathbf{y}}$ and the true output $\mathbf{y}$, the loss function is computed using the following formula:
$$\mathcal{L}(\theta_t) = \text{Loss}(\hat{\mathbf{y}}, \mathbf{y})$$

## Backward Propagation

### Overview

Backward propagation, also known as backpropagation, is the process of computing gradients of the loss function with respect to the network parameters. These gradients are then used to update the parameters in order to minimize the loss.

### Gradient Computation

For each layer $i$ in the network, backward propagation involves the following steps:

1. **Gradient of Loss Function with Respect to Output Layer**: Compute the gradient of the loss function with respect to the output layer activations $\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}}$.

2. **Gradient of Loss Function with Respect to Weights**: Use the chain rule to compute the gradient of the loss function with respect to the weights $\mathbf{W}^{(i)}$ for each layer $i$:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(i)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(i)}} \cdot \frac{\partial \mathbf{z}^{(i)}}{\partial \mathbf{W}^{(i)}}$$

3. **Gradient of Loss Function with Respect to Biases**: Similarly, compute the gradient of the loss function with respect to the biases $\mathbf{b}^{(i)}$ for each layer $i$:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(i)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(i)}} \cdot \frac{\partial \mathbf{z}^{(i)}}{\partial \mathbf{b}^{(i)}}$$

### Chain Rule

The chain rule is used to compute the gradients of the loss function with respect to the weights and biases. It allows us to decompose the overall gradient into smaller gradients that can be computed efficiently.

### Update Rule

Once the gradients have been computed, they are used to update the network parameters using an optimization algorithm such as gradient descent. The update rule for the weights is given by:
$$\mathbf{W}^{(i)}_{\text{new}} = \mathbf{W}^{(i)}_{\text{old}} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(i)}}$$
Where $\alpha$ is the learning rate, controlling the size of the updates.

# Conclusion

In conclusion, understanding the backpropagation algorithm is crucial for grasping the fundamentals of training neural networks. Backpropagation allows us to efficiently compute gradients of the loss function with respect to the network parameters, enabling iterative updates that minimize the loss and improve the model's performance. By decomposing the gradient computation using the chain rule and updating the parameters using optimization algorithms like gradient descent, we can effectively train complex neural networks to solve a wide range of tasks.

## Points to Remember

1. **Forward Propagation:**
   - Forward propagation computes the network's output given an input by passing it through the layers and applying activation functions.
   - Pre-activations are computed using weight matrices, activation vectors, and biases, followed by activation function application.
   - Output activation function transforms the final pre-activation into the network's prediction.

2. **Loss Computation:**
   - Loss function measures the error between predicted and true outputs and guides the training process.
   - Common loss functions include mean squared error, cross-entropy loss, and hinge loss.
   - Loss computation involves comparing predicted and true outputs using the chosen loss function.

3. **Backward Propagation:**
   - Backward propagation computes gradients of the loss function with respect to network parameters.
   - Gradient computation involves the chain rule to decompose gradients efficiently.
   - Gradients are used to update weights and biases, facilitating model improvement over iterations.

4. **Chain Rule:**
   - The chain rule allows the decomposition of complex gradients, simplifying the computation of gradients with respect to weights and biases.

5. **Update Rule:**
   - Update rule adjusts network parameters using gradients and a learning rate.
   - Learning rate controls the size of parameter updates, influencing the convergence and stability of the training process.

6. **Optimization Algorithms:**
   - Gradient descent is a common optimization algorithm used in conjunction with backpropagation for training neural networks.
   - Other optimization algorithms like Adam, RMSprop, and SGD with momentum offer variations for improved convergence and performance.

7. **Training Process:**
   - Training neural networks involves iterative forward and backward passes, adjusting parameters to minimize the loss function.
   - Effective training requires careful selection of hyperparameters, regularization techniques, and monitoring of model performance.
