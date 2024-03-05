# Adaptive Learning Rates

## Introduction

In the realm of deep learning, optimization algorithms play a crucial role in training neural networks. One such algorithm is gradient descent, which aims to minimize a loss function by iteratively updating the parameters of the network. However, traditional gradient descent algorithms often struggle with finding an appropriate learning rate that balances convergence speed and stability across different regions of the optimization landscape. This is where adaptive learning rates come into play.

## Adaptive Learning Rate Concepts

Adaptive learning rate algorithms dynamically adjust the learning rate during training based on the history of gradients and the current position in the optimization landscape. This adaptive behavior allows for faster convergence in regions with gentle gradients and more cautious updates in steep regions to prevent overshooting.

### Importance of Adaptive Learning Rates

Traditional gradient descent algorithms use a fixed learning rate, which may lead to suboptimal convergence behavior, especially in scenarios where the optimization landscape is highly variable. Adaptive learning rates address this issue by dynamically adjusting the learning rate based on the gradient's magnitude and direction, resulting in improved convergence and training efficiency.

## Neural Network Representation

To understand the application of adaptive learning rates, let's first revisit the representation of a neural network. Consider a neural network with $L$ layers, where each layer is composed of neurons. The input to the network is represented by $\mathbf{x}$, and the output is represented by $\mathbf{y}$. 

### Weight Matrices and Bias Vectors

At each layer $i$, the network applies a set of weights $\mathbf{W}^{(i)}$ and biases $\mathbf{b}^{(i)}$ to transform the input into a pre-activation vector $\mathbf{a}^{(i)}$. The pre-activation vector is then passed through an activation function $\mathbf{g}^{(i)}(\cdot)$ to produce the activation vector $\mathbf{h}^{(i)}$. This process is repeated for each subsequent layer until the final output is obtained.

### Loss Function and Optimization

During training, the network's parameters, including weights and biases, are updated iteratively to minimize a loss function $\mathcal{L}(\theta_t)$, where $\theta_t$ represents the parameters at iteration $t$. The optimization process involves computing the gradient of the loss function with respect to the parameters and adjusting the parameters in the direction that minimizes the loss.

## Derivation of Adaptive Learning Rates

Now, let's delve into the derivation of adaptive learning rates and their significance in optimizing neural networks.

### Derivative Calculation with Sparse Features

Consider a scenario where the input features $\mathbf{x}$ include sparse features, i.e., features that are often zero across many training instances. In such cases, the derivative of the loss function with respect to the weights corresponding to sparse features tends to be small due to the frequent occurrence of zero values.

Mathematically, let's denote the derivative of the loss function $\mathcal{L}$ with respect to a weight $\mathbf{W}_j^{(i)}$ as $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_j^{(i)}}$. If a feature $x_j$ is sparse, the derivative can be expressed as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_j^{(i)}} = \sum_{k=1}^{m} \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(i)}} \cdot x_j
$$

Where $m$ represents the total number of training instances, and $x_j$ is the value of the sparse feature.

### Impact of Sparse Features on Gradient Descent

Sparse features lead to sparse updates during gradient descent, as the derivatives associated with these features are small. Consequently, the weights corresponding to sparse features experience minimal changes during optimization, potentially impeding the network's ability to learn from these features effectively.

### Importance of Adaptive Learning Rates for Sparse Features

To address the issue of sparse updates for weights associated with sparse features, adaptive learning rates offer a solution. By dynamically adjusting the learning rate based on the sparsity of features, adaptive learning rates ensure that weights corresponding to sparse features receive meaningful updates, allowing the network to effectively leverage the information provided by these features.

## Implementing Adaptive Learning Rates

The implementation of adaptive learning rates involves designing algorithms that automatically adjust the learning rate based on the sparsity of features. This requires a systematic approach to ensure efficient optimization across millions of features without manual intervention.

### Mathematical Formulation

Let's denote the learning rate at iteration $t$ as $\eta_t$. To adaptively adjust the learning rate based on the sparsity of features, we can define a function $\eta_t = f(\mathbf{x}_t)$, where $\mathbf{x}_t$ represents the input data at iteration $t$.

One approach to defining the adaptive learning rate function is to incorporate a measure of feature sparsity into the learning rate calculation. For example, we can define $\eta_t$ as follows:

$$
\eta_t = \eta_0 \cdot \text{sparsity\_factor}(\mathbf{x}_t)
$$

Where $\eta_0$ represents the initial learning rate, and $\text{sparsity\_factor}(\mathbf{x}_t)$ is a function that quantifies the sparsity of the input data at iteration $t$.

### Benefits of Adaptive Learning Rates

By incorporating adaptive learning rates into the optimization process, neural networks can effectively leverage sparse features for improved performance. Adaptive learning rates ensure that weights associated with sparse features receive sufficient updates, allowing the network to learn meaningful representations from sparse data.

# AdaGrad

## Introduction

AdaGrad is a powerful optimization algorithm used in deep learning to adaptively adjust the learning rate during training. It addresses the challenge of selecting an appropriate learning rate for different features in the input data. This method ensures that features with frequent updates receive smaller learning rates, while features with sparse updates receive larger learning rates, leading to more effective and efficient training of neural networks.

## Update Rule for AdaGrad

The core idea behind AdaGrad is to adjust the learning rate for each feature based on its update history. This is achieved by maintaining a history of the squared gradients for each feature and dividing the learning rate by the square root of this history. Mathematically, the update rule for AdaGrad can be expressed as follows:

$$
\mathbf{v}_t = \mathbf{v}_{t-1} + \left( \frac{\partial \mathcal{L}(\theta_t)}{\partial \mathbf{W}_t} \right)^2
$$

$$
\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \frac{\partial \mathcal{L}(\theta_t)}{\partial \mathbf{W}_t}
$$

Where:

- $\mathbf{v}_t$ is the accumulated squared gradients.
- $\eta$ is the learning rate.
- $\epsilon$ is a small constant added to the denominator for numerical stability.
- $\frac{\partial \mathcal{L}(\theta_t)}{\partial \mathbf{W}_t}$ is the gradient of the loss function with respect to the weights at iteration $t$.
- $\mathbf{W}_t$ represents the weights at iteration $t$.

Similarly, the update rule for the bias terms $\mathbf{b}_t$ can be derived using the same principle.

## Implementation in Code

In code, the AdaGrad algorithm involves accumulating the squared gradients and updating the weights and biases accordingly. The update equations for weights and biases can be implemented as follows:

```python
v_W += (dW ** 2)
W -= (learning_rate / np.sqrt(v_W + eps)) * dW
```

```python
v_b += (db ** 2)
b -= (learning_rate / np.sqrt(v_b + eps)) * db
```

Where:

- `v_W` and `v_b` are the accumulated squared gradients for weights and biases, respectively.
- `dW` and `db` are the gradients of the loss function with respect to the weights and biases.
- `learning_rate` is the learning rate.
- `eps` is a small constant for numerical stability.

## Experimental Results and Analysis

In an experiment, AdaGrad was applied to training data with both sparse and dense features. It demonstrated the ability to make proportionate movements in the direction of sparse features, despite their small updates. However, one observation was that as the training progressed, AdaGrad's effective learning rate decreased significantly, potentially causing slow convergence near the minimum.

## Visual Analysis of AdaGrad Behavior

Visual analysis of AdaGrad's behavior revealed that it was able to move proportionately in the direction of both sparse and dense features. However, as the training progressed, AdaGrad's effective learning rate decreased exponentially due to the accumulation of update history.

## Challenges and Limitations

While AdaGrad successfully adapted the learning rate for both sparse and dense features, it faced challenges as training progressed. The effective learning rate for dense features became so small that it hindered movement in the direction of sparse features, leading to slower convergence. The accumulation of update history posed a challenge near the minimum, where gradients became small but the history remained large, causing the effective learning rate to decrease excessively.

## Potential Improvements

One potential improvement could be to explore variations of AdaGrad that address the issue of excessively small effective learning rates. Adding a momentum term to AdaGrad could potentially combine the advantages of momentum-based algorithms with adaptive learning rates, improving convergence speed and performance. However, further research and experimentation are needed to explore these possibilities and address the limitations of AdaGrad effectively.

# RMSprop

## Introduction

In the realm of deep learning, optimization algorithms play a crucial role in training neural networks effectively. One such algorithm is RMSprop, short for Root Mean Square Propagation, which addresses some of the limitations of previous optimization methods like AdaGrad. In this lecture module, we delve into the intricacies of RMSprop, its formulation, and its behavior during the training process.

## Motivation for RMSprop

The motivation behind RMSprop stems from the need to address the aggressive decay of learning rates in optimization algorithms as the training progresses. In AdaGrad, for instance, the denominator in the update rule accumulates the squares of past gradients, causing the learning rate to diminish rapidly, especially for frequently updated parameters. This phenomenon inhibits the convergence of the optimization process, leading to suboptimal solutions. 

## Formulation of RMSprop

To mitigate the rapid growth of the denominator in AdaGrad, RMSprop introduces a scaling mechanism by modifying the update rule. Instead of accumulating the squares of gradients indiscriminately, RMSprop employs an exponentially decaying average of past squared gradients. This is achieved by introducing a decay factor, typically denoted as $\beta$, which controls the rate at which the history is accumulated. The modified update rule for the denominator $v_t$ in RMSprop is given by:

$$ v_t = \beta v_{t-1} + (1 - \beta) (\nabla \mathcal{L})^2 $$

where:

- $v_t$ represents the accumulated history of squared gradients at iteration $t$.
- $\beta$ is a hyperparameter controlling the exponential decay rate.
- $\nabla \mathcal{L}$ denotes the gradient of the loss function.

## Key Insights into RMSprop

### Exponentially Decaying Average

The crux of RMSprop lies in the use of an exponentially decaying average for accumulating the history of squared gradients. This approach ensures that the denominator $v_t$ grows less aggressively compared to AdaGrad, thereby stabilizing the effective learning rate.

### Control over Learning Rate Decay

By scaling down the growth of the denominator with the help of the decay factor $\beta$, RMSprop prevents the rapid decline of the effective learning rate. This control over the learning rate decay allows for smoother convergence during optimization.

### Comparison with AdaGrad

In contrast to AdaGrad, where the denominator accumulates gradients without any decay, RMSprop offers a more controlled approach by incorporating an exponentially decaying average. This modification alleviates the issue of overly aggressive learning rate decay encountered in AdaGrad.

## Behavior of RMSprop during Training

### Effect on Learning Rate

During the training process, RMSprop dynamically adjusts the effective learning rate based on the magnitude of gradients encountered. In regions with steep gradients, the learning rate decreases gradually to prevent overshooting, while in flatter regions, it increases to expedite convergence.

### Sensitivity to Initial Learning Rate

One notable aspect of RMSprop is its sensitivity to the initial learning rate ($\eta_0$). The choice of $\eta_0$ can significantly impact the convergence behavior of the algorithm, leading to variations in convergence speed and stability.

### Oscillation Phenomenon

In some scenarios, RMSprop may exhibit oscillations around the minima during optimization. These oscillations stem from the interplay between the learning rate and the curvature of the loss surface. If the learning rate becomes constant and the curvature allows for symmetric oscillations, the optimization process may oscillate between different points on the loss surface.

## Addressing Sensitivity to Initial Learning Rate

### Adaptive Learning Rate Adjustment

To mitigate the sensitivity to the initial learning rate, researchers have proposed adaptive techniques that dynamically adjust the learning rate during training. These methods aim to alleviate the reliance on manually tuning the initial learning rate, thereby improving the robustness of optimization algorithms like RMSprop.

### Experimental Exploration

Empirical studies have shown that the choice of initial learning rate ($\eta_0$) can significantly impact the convergence behavior of RMSprop. Researchers often conduct experiments with different values of $\eta_0$ to determine the optimal setting for specific datasets and network architectures.

# AdaDelta

## Introduction
In the domain of deep learning, optimization algorithms play a crucial role in training neural networks effectively. One such algorithm is AdaDelta, which is designed to address challenges such as choosing an appropriate learning rate and dealing with varying magnitudes of gradients during training. This algorithm dynamically adapts the learning rate based on past gradients, allowing for smoother convergence and improved performance. In this section, we delve into the details of the AdaDelta algorithm, its key components, and its application in optimizing neural network parameters.

## Overview of AdaDelta
AdaDelta is an extension of the RMSprop optimization algorithm, which aims to mitigate its dependency on an initial learning rate. Unlike traditional methods that require manual tuning of hyperparameters like the learning rate, AdaDelta automatically adjusts the learning rate during training based on past gradients and accumulated updates.

## Mathematical Formulation
Let's define some key variables and equations used in AdaDelta:

**Variables:**

- $\mathbf{u}_t$: Velocity at iteration $t$
- $v_t$: Accumulated history of squared gradients at iteration $t$
- $\beta$: Hyperparameter controlling the exponential decay rate

**Equations:**

1. **Update Rule**:
   $$ \Delta \mathbf{W}_t = - \frac{\sqrt{\mathbf{u}_{t-1} + \epsilon}}{\sqrt{v_t + \epsilon}} \cdot \nabla \mathcal{L} $$
   
2. **Update Velocity**:
   $$ \mathbf{u}_t = \beta \cdot \mathbf{u}_{t-1} + (1 - \beta) \cdot (\Delta \mathbf{W}_t)^2 $$
   
3. **Parameter Update**:
   $$ \mathbf{W}_{t+1} = \mathbf{W}_t + \Delta \mathbf{W}_t $$

## Key Concepts

### Exponential Moving Averages
AdaDelta utilizes exponential moving averages to compute the update velocity and accumulated history of squared gradients. This involves maintaining a running average of past gradients and squared gradients, weighted by the decay factor $\beta$. By doing so, AdaDelta can adaptively adjust the learning rate based on the magnitude and variance of gradients encountered during training.

### Ratio of Updates
The AdaDelta algorithm calculates the update as a ratio of two variables: $\mathbf{u}_t$ and $v_t$. This ratio serves as a scaling factor for the gradient, allowing AdaDelta to effectively modulate the learning rate based on the historical behavior of gradients.

### Adaptive Learning Rate
Unlike traditional optimization algorithms that rely on a fixed learning rate, AdaDelta dynamically adjusts the learning rate based on the accumulated history of gradients. This adaptive nature enables AdaDelta to navigate complex optimization landscapes more efficiently and converge to optimal solutions with fewer iterations.

## Algorithm Workflow
Now, let's outline the step-by-step workflow of the AdaDelta algorithm:

1. **Initialization**:
   - Initialize parameters and variables, including $\mathbf{W}$, $\mathbf{u}$, and $v$.
   - Set hyperparameters such as $\beta$ and $\epsilon$.

2. **Compute Gradient**:
   - Calculate the gradient of the loss function with respect to the model parameters ($\nabla \mathcal{L}$).

3. **Update Velocity**:
   - Update the velocity ($\mathbf{u}_t$) using the current gradient and the decay factor $\beta$.

4. **Compute Update**:
   - Compute the update ($\Delta \mathbf{W}_t$) using the ratio of $\mathbf{u}_t$ and $v_t$.

5. **Parameter Update**:
   - Update the model parameters ($\mathbf{W}$) using the computed update.

6. **Repeat**:
   - Iterate through steps 2-5 for multiple epochs or until convergence criteria are met.

## Advantages of AdaDelta
AdaDelta offers several advantages over traditional optimization algorithms:

1. **Automatic Learning Rate Adjustment**:
   - AdaDelta eliminates the need for manually tuning the learning rate by adapting it dynamically based on past gradients.

2. **Improved Convergence**:
   - By adjusting the learning rate according to the historical behavior of gradients, AdaDelta can converge more smoothly and efficiently.

3. **Robustness to Hyperparameters**:
   - AdaDelta's reliance on only a few hyperparameters, such as $\beta$, makes it more robust and easier to use compared to algorithms with additional tuning parameters.

# Adam

Adam, short for Adaptive Moments, combines elements of RMSprop and momentum-based optimization techniques to achieve adaptive learning rates. The algorithm maintains exponentially weighted averages of past gradients and squared gradients to adjust the effective learning rate for each parameter.

### Components of Adam Algorithm

Adam algorithm consists of the following key components:

1. **Exponentially Weighted Averages**: Adam maintains two exponentially weighted moving averages: $\mathbf{m}_t$ for the gradients and $\mathbf{v}_t$ for the squared gradients.

2. **Bias Correction**: Adam incorporates bias correction terms to mitigate the initialization bias, ensuring smoother updates during the initial training phases.

3. **Effective Learning Rate**: The effective learning rate in Adam is computed based on the moving averages of gradients and squared gradients, adjusted by bias correction factors.

### Update Equations for Adam

The update equations for Adam algorithm are as follows:

$$
\begin{align*}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla \mathcal{L}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla \mathcal{L}_t)^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
\mathbf{W}_t &= \mathbf{W}_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{align*}
$$

where:

- $\beta_1$ and $\beta_2$ are hyperparameters controlling the exponential decay rates.
- $\nabla \mathcal{L}_t$ denotes the gradient of the loss function at iteration $t$.
- $\eta$ is the learning rate.
- $\epsilon$ is a small constant to prevent division by zero.

### Rationale behind Bias Correction

The bias correction term in Adam addresses the initialization bias observed in the early stages of training. By dividing the moving averages by the bias correction factors, Adam ensures that the effective learning rate remains stable across different iterations, preventing erratic updates during the initial training phase.

## Comparison with Other Adaptive Algorithms

Adam algorithm exhibits favorable convergence properties compared to other adaptive algorithms such as AdaDelta and RMSprop. By incorporating both momentum and adaptive learning rate mechanisms, Adam achieves faster convergence while avoiding the learning rate decay issues encountered in RMSprop.

### Experimental Results

Empirical studies have demonstrated the superior performance of Adam in terms of convergence speed and generalization ability. By dynamically adjusting the learning rate based on past gradients and squared gradients, Adam effectively navigates the loss landscape, leading to faster convergence and improved model performance.

# LP Norms and Optimization

## Introduction

In deep learning, optimization algorithms play a crucial role in training neural networks efficiently. Understanding different norms and their implications on optimization is essential for designing effective optimization techniques. In this discussion, we delve into LP Norms and their significance in optimization algorithms, particularly focusing on the Adam optimizer.

## LP Norms

LP Norm is a mathematical concept used to measure the size of a vector in a space. It is defined by the following formula:

$$
\| \mathbf{x} \|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}
$$

where $\mathbf{x}$ is the input vector, $p$ is a parameter, and $n$ is the dimensionality of the vector.

### L2 Norm

The L2 Norm, also known as the Euclidean Norm, is a special case of the LP Norm where $p = 2$. It is calculated by taking the square root of the sum of squares of the vector components:

$$
\| \mathbf{x} \|_2 = \sqrt{\sum_{i=1}^{n} |x_i|^2}
$$

The L2 Norm is widely used in deep learning for regularization and optimization purposes.

### L Infinity Norm

The L Infinity Norm, denoted as $\| \mathbf{x} \|_{\infty}$, represents the maximum absolute value of the vector components:

$$
\| \mathbf{x} \|_{\infty} = \max_{i} |x_i|
$$

It simplifies computations and is particularly useful in scenarios where the maximum magnitude of the elements is of interest.

## Optimization with LP Norms

Optimization algorithms in deep learning often involve computing gradients and updating model parameters iteratively. The choice of norm used in these algorithms can have significant implications on convergence and performance.

### Adam Optimizer with Exponentially Weighted L2 Norm

The Adam optimizer is a popular choice for training neural networks due to its adaptive learning rate mechanism. It incorporates an exponentially weighted L2 Norm of gradients to adaptively adjust the learning rate for each parameter.

The update rule for the Adam optimizer involves maintaining two exponentially decaying moving averages: $m_t$ for the first moment (mean) and $v_t$ for the second moment (uncentered variance) of the gradients.

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla \mathcal{L}_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla \mathcal{L}_t)^2
$$

where $\beta_1$ and $\beta_2$ are hyperparameters controlling the exponential decay rates, $\nabla \mathcal{L}_t$ is the gradient of the loss function at iteration $t$.

### Adam Max: Introducing Max Norm

In the context of the Adam optimizer, the use of L2 Norm for computing the gradient's magnitude may lead to numerical instability, especially when dealing with large values of $p$. To address this issue, we explore the possibility of using the L Infinity Norm (Max Norm) instead.

The Max Norm, defined as $\| \mathbf{x} \|_{\infty} = \max_{i} |x_i|$, simplifies to selecting the maximum absolute value from the vector components.

### Benefits of Using Max Norm

1. **Simplicity**: The Max Norm computation is straightforward and does not involve complex mathematical operations.
  
2. **Stability**: Max Norm avoids numerical instability issues associated with large values of $p$ in LP Norms, making it a robust choice for optimization algorithms.

### Update Rule for Adam Max

The update rule for Adam Max, a variant of the Adam optimizer using Max Norm, is derived by replacing the L2 Norm computation with the Max Norm for computing the second moment $v_t$. 

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \| \nabla \mathcal{L}_t \|_{\infty}^2
$$

This modification simplifies the computation and enhances the stability of the optimization process.

## Comparison with L2 Norm

To understand the practical implications of using Max Norm in optimization, let's compare its performance with the traditional L2 Norm approach.

### Scenario 1: Sparse Gradients

In scenarios where gradients alternate between high and zero values, the Max Norm maintains a more consistent learning rate compared to the L2 Norm. This stability ensures smoother convergence during training, especially when dealing with sparse features.

### Scenario 2: Zero Inputs

When encountering zero inputs, the Max Norm prevents unnecessary fluctuations in the learning rate. Unlike the L2 Norm, which may amplify changes even with zero gradients, the Max Norm remains stable and preserves the learning rate effectively.

# NADAM

## Introduction

Nesterov Accelerated Gradient Descent (NAG) is an optimization algorithm used in training neural networks. It is an extension of the standard momentum-based gradient descent method. The key idea behind NAG is to improve upon the momentum-based approach by incorporating the Nesterov's accelerated gradient (NAG) concept, also known as the lookahead effect. In this module, we explore how to integrate Nesterov Accelerated Gradient Descent into the Adam optimizer.

## Rewriting NAG Equations

### Original NAG Update Rule

The original NAG update rule involves computing the gradient at a lookahead value and then updating the parameters based on this lookahead gradient. This approach involves cumbersome computations and redundant calculations.

### Simplifying NAG Equations

To simplify the NAG equations and integrate them into the Adam optimizer, we need to rewrite the update rule in a more compact and efficient manner. The goal is to eliminate redundant computations and express all equations in terms of the current time step ($t$) and the next time step ($t + 1$).

## Modified NAG Equations

### Update Rule for Nesterov Accelerated Gradient Descent

The update rule for Nesterov Accelerated Gradient Descent (NAG) involves the following steps:

1. Compute the gradient at the current parameter values.
2. Compute the lookahead gradient at the next parameter values using the gradient computed in step 1.
3. Update the parameters using a combination of the current gradient and the lookahead gradient.

### Mathematical Formulation of NAG

The Nesterov Accelerated Gradient Descent update rule can be expressed as follows:

$$
\mathbf{u}_{t+1} = \beta \mathbf{u}_t + \eta \nabla \mathcal{L}(\theta_t - \beta \mathbf{u}_t)
$$

$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla \mathcal{L}_t)^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1}} + \epsilon} \left( \beta_1 \mathbf{u}_{t+1} + (1 - \beta_1) \nabla \mathcal{L}_t \right)
$$

where:

- $\mathbf{u}_t$ represents the velocity at iteration $t$.
- $v_t$ represents the accumulated history of squared gradients at iteration $t$.
- $\beta$ is a hyperparameter controlling the exponential decay rate for the velocity.
- $\beta_1$ and $\beta_2$ are hyperparameters controlling the exponential decay rates for the velocity and squared gradients, respectively.
- $\eta$ is the learning rate.
- $\epsilon$ is a small constant to prevent division by zero.

## Practical Considerations and Conclusion

### Choosing the Optimizer

In practical applications, choosing the right optimizer is crucial for achieving good performance in training neural networks. While there are various optimization algorithms available, Adam optimizer is widely used due to its effectiveness in many scenarios. However, other variants such as Nesterov Accelerated Gradient Descent (NAG) can also be considered, especially when dealing with specific optimization challenges.

### Learning Rate Schedules

In addition to selecting the optimizer, tuning the learning rate schedule is another important aspect of training deep learning models. Proper adjustment of the learning rate can significantly impact the convergence and stability of the optimization process. Experimenting with different learning rate schedules and monitoring the training process can help determine the optimal settings for achieving desired performance.

# Learning Rate Schedules

In deep learning, the choice of learning rate schedule plays a crucial role in optimizing neural network models. This section explores various learning rate schemes, including epoch-based and adaptive approaches, as well as cyclic learning rate schedules such as cyclical and cosine annealing.

## Introduction to Learning Rate Schedules

In neural network training, the learning rate ($\eta$) determines the step size during gradient descent optimization. Choosing an appropriate learning rate schedule can significantly impact the convergence and performance of the model. Different learning rate schedules adjust the learning rate over time to facilitate effective optimization.

## Epoch-Based Learning Rate Schemes

Epoch-based learning rate schemes adjust the learning rate based on the number of training epochs. Common approaches include step decay and exponential decay.

### Step Decay

In step decay, the learning rate is reduced by a factor ($\gamma$) after a fixed number of epochs ($\tau$). Mathematically, it can be expressed as:

$$ \eta_t = \eta_0 \times \gamma^{\lfloor \frac{t}{\tau} \rfloor} $$

where $\eta_t$ is the learning rate at iteration $t$, $\eta_0$ is the initial learning rate, $\gamma$ is the decay factor, and $\tau$ is the step size.

### Exponential Decay

Exponential decay reduces the learning rate exponentially over time. The learning rate at iteration $t$ is given by:

$$ \eta_t = \eta_0 \times e^{-\lambda t} $$

where $\lambda$ controls the rate of decay.

## Adaptive Learning Rate Schemes

Adaptive learning rate schemes dynamically adjust the learning rate based on past gradients or other parameters. Examples include Adagrad, RMSProp, ADA Delta, Adam, and Adamax.

### Adagrad

Adagrad adapts the learning rate for each parameter based on the magnitude of its gradients. It scales the learning rate inversely proportional to the square root of the sum of squared gradients.

$$ \eta_t = \frac{\eta_0}{\sqrt{v_t + \epsilon}} $$

where $v_t$ represents the accumulated history of squared gradients at iteration $t$ and $\epsilon$ is a small constant to prevent division by zero.

### RMSProp

RMSProp improves upon Adagrad by using a moving average of squared gradients for scaling the learning rate. It addresses the diminishing learning rate problem in Adagrad by using a decay rate $\beta$.

$$ v_t = \beta v_{t-1} + (1 - \beta) (\nabla \mathcal{L}_t)^2 $$

where $\nabla \mathcal{L}_t$ denotes the gradient of the loss function at iteration $t$.

### ADA Delta

ADA Delta further enhances RMSProp by replacing the learning rate with the root mean square (RMS) of parameter updates.

$$ \eta_t = \sqrt{\frac{v_{t-1} + \epsilon}{v_t + \epsilon}} $$

### Adam

Adam combines the advantages of both RMSProp and momentum optimization. It maintains two moving averages for gradients and squared gradients.

$$ \mathbf{u}_t = \beta_1 \mathbf{u}_{t-1} + (1 - \beta_1) \nabla \mathcal{L}_t $$
$$ v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla \mathcal{L}_t)^2 $$
$$ \eta_t = \frac{\eta}{\sqrt{v_t + \epsilon}} $$

where $\beta_1$ and $\beta_2$ are hyperparameters controlling the exponential decay rates.

### Adamax

Adamax is a variant of Adam that replaces the $L_2$ norm with the $L_{\infty}$ norm.

## Cyclic Learning Rate Schedules

Cyclic learning rate schedules alternate between increasing and decreasing the learning rate over a predefined range.

### Triangular Schedule

The triangular schedule cyclically increases the learning rate from a minimum to maximum value and back. It helps escape saddle points by periodically increasing the learning rate.

$$ \eta_t = \eta_{\text{min}} + (\eta_{\text{max}} - \eta_{\text{min}}) \times \text{max}(0, 1 - |\frac{T}{\mu} - 2 \lfloor \frac{T}{\mu} \rfloor - 1|) $$

where $\mu$ is the period of the cycle.

### Cosine Annealing

Cosine annealing smoothly decreases the learning rate using a cosine function. It converges faster compared to fixed learning rates.

$$ \eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) (1 + \cos(\frac{T}{T_{\text{max}}} \pi)) $$

where $T$ is the current epoch and $T_{\text{max}}$ is the restart interval.

### Warm Restart

Warm restart involves quickly jumping from the minimum to maximum learning rate and then decaying. It is popular in Transformer architectures.

