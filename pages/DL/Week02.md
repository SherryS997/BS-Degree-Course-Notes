---
title: "Deep Learning Foundations: From Boolean Functions to Universal Approximation"
---


# Boolean Functions and Linear Separability

## Introduction

Boolean functions, fundamental to computational logic, pose challenges when it comes to their linear separability. The perceptron learning algorithm, known for its guarantees with linearly separable data, encounters limitations when dealing with certain boolean functions. This module delves into the intricacies of these functions and explores the concept of linear separability.

## XOR Function Analysis

### XOR Function Definition

The XOR function, denoted as $f(x_1, x_2)$, outputs 1 when exactly one of its inputs is 1. It follows the logic:
$$f(0,0) \rightarrow 0, \, f(0,1) \rightarrow 1, \, f(1,0) \rightarrow 1, \, f(1,1) \rightarrow 0$$

### Perceptron Implementation Challenges

Attempting to implement XOR using a perceptron leads to a set of four inequalities. These conditions, when applied to weights ($w_0, w_1, w_2$), cannot be simultaneously satisfied. Geometrically, this signifies the inability to draw a line that separates positive and negative points in the XOR function.

## Implications for Real-World Data

Real-world data often deviates from the assumption of linear separability. For instance, individuals with similar characteristics may exhibit diverse preferences, challenging the effectiveness of linear decision boundaries.

## Network of Perceptrons

Recognizing the limitations of a single perceptron in handling non-linearly separable data, a proposed solution involves using a network of perceptrons. This approach aims to extend the capability of handling complex, non-linearly separable boolean functions.

## Boolean Functions from N Inputs

Boolean functions with $n$ inputs offer a wide range of possibilities, such as AND, OR, and others. The total number of boolean functions from $n$ inputs is given by $2^{2^n}$. The discussion extends to the linear separability of these boolean functions.

## Challenge of Non-Linear Separability

Out of the $2^{2^n}$ boolean functions, some are not linearly separable. The precise count of non-linearly separable functions remains an unsolved problem, highlighting the need for robust methods capable of handling such cases.

# Multi-Layer Perceptrons (MLPs) and Boolean Function Representation

## Introduction to Multi-Layer Perceptrons

Multi-Layer Perceptrons (MLPs) constitute a pivotal advancement in artificial neural networks. These networks boast a layered architecture, each layer serving a distinct role in processing information.

### Layers in an MLP

1. **Input Layer:**
   - Comprising nodes representing input features ($x_1, x_2, ..., x_n$).
   
2. **Hidden Layer:**
   - Features multiple perceptrons introducing non-linearities to the network.
   
3. **Output Layer:**
   - Houses a single perceptron providing the final network output.

### Weights and Bias

1. **Connection Characteristics:**
   - Weights ($w$) and a bias term ($w_0$) define the connections between nodes.
   
2. **Weighted Sum and Activation:**
   - The weighted sum of inputs, combined with the bias, influences perceptron activation.

## Representation of Boolean Functions in MLPs

### Network Structure for Boolean Functions

1. **Hidden Layer Configuration:**
   - For a boolean function with $n$ inputs, the hidden layer consists of $2^n$ perceptrons.
   
2. **Weight and Bias Adjustment:**
   - Weights and biases are adjusted to meet boolean logic conditions for accurate function representation.

### Boolean Function Implementation

1. **Perceptron Activation Conditions:**
   - Each perceptron in the hidden layer selectively fires based on specific input combinations.
   
2. **XOR Function Illustration:**
   - Using the XOR function as an example, conditions on weights ($w_1, w_2, w_3, w_4$) are established for faithful representation.

3. **Extension to $n$ Inputs:**
   - Generalizing the approach to $n$ inputs involves $2^n$ perceptrons in the hidden layer.
   - Conditions for output layer weights are derived to ensure accurate representation.

## Representation Power and Implications

### Representation Power Theorem

1. **Theorem Statement:**
   - Any boolean function of $n$ inputs can be precisely represented by an MLP.
   
2. **Suggested MLP Structure:**
   - An MLP with $2^n$ perceptrons in the hidden layer and 1 perceptron in the output layer is deemed sufficient.

### Practical Considerations

1. **Challenges with Growing $n$:**
   - The exponential increase in perceptrons as $n$ grows poses practical challenges.
   
2. **Real-World Applications:**
   - Managing and computing with a large number of perceptrons may be challenging in practical applications.

# Introduction to Sigmoid Neurons and the Sigmoid Function

## Transition from Perceptrons to Sigmoid Neurons

### Binary Output Limitation
Perceptrons, governed by binary output based on the weighted sum of inputs exceeding a threshold, exhibit a binary decision boundary. This rigid characteristic proves restrictive in scenarios where a more gradual decision-making process is preferred.

### Real-Valued Inputs and Outputs
The shift towards sigmoid neurons arises in the context of addressing arbitrary functions $Y = f(X)$, wherein $X \in \mathbb{R}^n$ and $Y \in \mathbb{R}$. This entails the consideration of real numbers for both inputs and outputs. Examples include predicting oil quantity based on salinity, density, pressure, temperature, and marine diversity, as well as determining bank interest rates considering factors like salary, family size, previous loans, and defaults.

## Objective

The primary objective is to construct a neural network capable of accurately approximating or representing real-valued functions, ensuring the proximity of the network's output to actual values present in the training data.

## Introduction to Sigmoid Neurons

### Sigmoid Function
Sigmoid neurons employ the sigmoid function (logistic function) to introduce smoothness in decision-making. Mathematically represented as:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z$ denotes the weighted sum of inputs.

### Sigmoid Function Properties
1. As $z$ tends to positive infinity: $\lim_{{z \to \infty}} \sigma(z) = 1$
2. As $z$ tends to negative infinity: $\lim_{{z \to -\infty}} \sigma(z) = 0$
3. At $W^T X = 0$: $\sigma(0) = \frac{1}{2}$

The sigmoid function transforms outputs into the range [0, 1], facilitating a probabilistic interpretation.

### Comparison with Perceptron
Contrasting with the perceptron function, the sigmoid function exhibits smoothness and continuity. The perceptron function lacks differentiability at the abrupt change in value, whereas the sigmoid function is differentiable.

## Importance of Differentiability
Differentiability holds paramount importance for various machine learning algorithms, particularly in derivative-related operations. The application of calculus in neural network training and optimization is streamlined by the differentiability of the sigmoid neuron's activation function.

# Supervised Machine Learning Setup

## Overview
In the realm of supervised machine learning, the fundamental objective is to comprehend the intricate structure of the setup, which encompasses various components crucial for effective model training. These components include the dataset, model representation, the learning algorithm, and the definition of an objective function.

## Components

### Data Representation
The dataset, denoted as $(x_i, y_i)$, is pivotal to the learning process. Here, $x_i$ signifies an $m$-dimensional input vector, while $y_i$ represents a real-valued output associated with the given input. The dataset essentially comprises a collection of such input-output pairs.

### Model Assumption
A critical assumption in this paradigm is that the output $y$ is contingent upon the input $x$, expressed as $y = f(x)$. However, the specific form of the function $f$ remains elusive, prompting the need for learning algorithms to discern it from the provided data.

#### Learning Algorithm
The learning algorithm employed in this context is the Gradient Descent algorithm. This iterative approach facilitates the adjustment of model parameters, ensuring a continuous refinement of the model's approximation.

### Objective Function (Loss Function)
Central to the learning process is the formulation of an objective function, commonly referred to as the Loss Function. Mathematically, it is defined as follows:

$$\mathcal{L}(\theta) = \sum_{i=1}^{n} \text{Difference}(y_{\hat{i}}, y_i)$$

Here, $\theta$ denotes the parameters of the model, and $\text{Difference}(y_{\hat{i}}, y_i)$ quantifies the dissimilarity between the predicted ($y_{\hat{i}}$) and actual ($y_i$) values.

## Objective Function Details

### Difference Function (Squared Error Loss)
The Difference Function, an integral component of the Loss Function, is expressed as:

$$\text{Difference}(\hat{y}, y) = (\hat{y} - y)^2$$

The squaring operation is implemented to ensure that both positive and negative errors contribute to the overall loss without canceling each other out.

## Analogy with Learning Trigonometry

### Training Phase
Analogous to mastering a chapter in a textbook, the training phase strives for zero or minimal errors on the content encapsulated within the training dataset.

### Validation Phase
Resembling the solving of exercises at the end of a chapter, the validation phase allows for revisiting and enhancing comprehension based on additional exercises.

### Test Phase (Exam)
The test phase simulates a real-world scenario where the model encounters new data. Unlike the training and validation phases, there is no opportunity for revisiting and refining the learned information.


# Learning Parameters: (Infeasible) guess work

## Introduction

Supervised machine learning involves the development of algorithms to learn parameters for a given model. This process aims to minimize the difference between predicted and actual values using a defined objective function. In this context, we explore a simplified model with one input, connected by weight ($w$), and a bias ($b$).

### Model Representation

The model is represented as $f(\mathbf{x}) = -w \mathbf{x} + b$, where $\mathbf{x}$ is the input vector. The task is to determine an algorithm that learns the optimal values for $w$ and $b$ using training data.

### Training Objective

The training objective involves minimizing the average difference between predicted values ($f(\mathbf{x})$) and actual values ($y$) over all training points. The process requires finding the optimal $w$ and $b$ values that achieve this minimum loss.

## Training Data

The training data consists of pairs $(\mathbf{x}, y)$, where $\mathbf{x}$ represents the input, and $y$ corresponds to the output. The loss function is defined as the average difference between predicted and actual values across all training points.

## Loss Function

The loss function is expressed as:

$$\mathcal{L}(w, b) = \frac{1}{N} \sum_{i=1}^{N} \left| f(\mathbf{x}_i) - y_i \right|$$

Here, $N$ is the number of training points, $\mathbf{x}_i$ is the input for the $i$-th point, and $y_i$ is the corresponding actual output.

## Trial-and-Error Approach

To illustrate the concept, a trial-and-error approach is employed initially. Random values for $w$ and $b$ are chosen, and the loss is calculated. Adjustments are made iteratively to minimize the loss. This process involves systematically changing $w$ and $b$ values until an optimal solution is found.

### Visualization with Error Surface

A 3D surface plot is used to visualize the loss in the $w-b$ plane. This plot aids in identifying regions of low and high loss. However, the impracticality of exhaustively exploring this surface for large datasets is acknowledged due to computational constraints.

# Learning Parameters: Taylor series approximation

## Introduction
The transcript delves into the intricacies of parameter optimization, focusing on the goal of efficiently traversing the error surface to reach the minimum error. The parameters of interest, denoted as $\theta$, are expressed as vectors, specifically encompassing $W$ and $B$ in the context of a toy network.

## Update Rule with Conservative Movement
The update rule for altering $\theta$ entails a meticulous adjustment of the parameters. The process involves taking a measured step, determined by a scalar $\eta$, in the direction of $\Delta\theta$, which encapsulates the parameter changes. This introduces a level of conservatism in the parameter adjustments, promoting stability in the optimization process.

## Taylor Series for Function Approximation
### Overview
The lecture introduces the Taylor series, a powerful mathematical tool for approximating functions that exhibit continuous differentiability. This method enables the representation of a function through polynomials, allowing for varying degrees of precision in the approximation.

### Linear Approximation
Linear approximation entails the establishment of a tangent line at a specific point on the function. This approach provides an initial approximation, and the accuracy is contingent on the chosen neighborhood size, denoted as $\varepsilon$.

### Quadratic and Higher-Order Approximations
Quadratic and higher-order approximations extend the accuracy of the approximation by incorporating additional terms. The lecture underscores the importance of selecting a small neighborhood for these approximations to maintain efficacy.

## Extending Concepts to Multiple Dimensions
The discussion expands to functions with two variables, exemplifying how linear and quadratic approximations operate in multidimensional spaces. The lecture underscores the critical role of confined neighborhoods ($\varepsilon$) in ensuring the precision of the Taylor series method across varying dimensions.

# Gradient Descent: Mathematical Foundation

## Introduction
In the realm of optimization for machine learning models, the process of iteratively updating parameters to minimize a loss function is a fundamental concept. One key technique employed in this context is **gradient descent**. This discussion delves into the intricate mathematical foundations underpinning gradient descent, focusing on the decision criteria for parameter updates and the optimization of the update vector.

## Taylor Series Expansion

### Objective
The overarching objective is to determine an optimal change in parameters, denoted as $\Delta\theta$ (represented as $\mathbf{U}$), to minimize the loss function $\mathcal{L}(\theta)$.

### Linear Approximation
Utilizing the Taylor series, the loss function at a nearby point $\theta + \Delta\theta$ is approximated linearly as:
$$\mathcal{L}(\theta + \Delta\theta) \approx \mathcal{L}(\theta) + \eta\mathbf{U}^T\nabla \mathcal{L}(\theta)$$
Here, $\eta$ is a small positive scalar, ensuring a negligible difference.

## Mathematical Aspects of Gradient Descent

### Gradient
The gradient $\nabla \mathcal{L}(\theta)$ is introduced as a vector comprising partial derivatives of the loss function with respect to its parameters. For a function $y = W^2 + B^2$ with two variables, the gradient is expressed as $[2W, 2B]$.

### Second Order Derivative (Hessian)
The concept of the Hessian matrix, representing the second-order derivative, is introduced. This matrix provides insights into the curvature of the loss function. In the case of a two-variable function, the Hessian is illustrated as a $2\times2$ matrix.

## Decision Criteria for Parameter Updates

### Linear Approximation and Criteria
The focus shifts to linear approximation, with higher-order terms neglected when $\eta$ is small. The decision criteria for a favorable parameter update is based on the condition:
$$\eta\mathbf{U}^T\nabla \mathcal{L}(\theta) < 0$$

## Optimization of Update Vector $\mathbf{U}$

### Angle $\beta$ and Cosine
Optimizing the update vector involves considering the angle $\beta$ between $\mathbf{U}$ and the gradient vector. The cosine of $\beta$, denoted as $\cos(\beta)$, is explored, and its range is discussed.

### Optimal Update for Maximum Descent
In the pursuit of maximum descent, the optimal scenario arises when $\cos(\beta) = -1$, indicating that the angle $\beta$ is 180 degrees, signifying movement in the direction opposite to the gradient vector. This aligns with the well-known rule in gradient descent: "Move in the direction opposite to the gradient."

# Gradient Descent for Sigmoid Neuron Optimization

## Overview

In the pursuit of optimizing the parameters of a sigmoid neuron, the lecture primarily delves into the application of the gradient descent algorithm. The primary objective is to minimize the associated loss function, thereby identifying optimal values for the neuron's weights ($W$) and bias ($B$).

## Key Concepts

### 1. Gradient Descent Rule

The **gradient descent rule** serves as an iterative optimization technique employed to minimize the loss function. The core update rule is defined as follows:

$$
W = W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}, \quad B = B - \eta \cdot \frac{\partial \mathcal{L}}{\partial B}
$$

This iterative process aims to iteratively refine the parameters ($W$ and $B$) based on the computed partial derivatives of the loss function.

### 2. Derivative Computation

#### 2.1 Derivative of Loss with Respect to $W$

The **partial derivative of the loss function with respect to weights ($\frac{\partial \mathcal{L}}{\partial W}$)** is computed through the application of the chain rule. In the context of the sigmoid function, the derivative is obtained as follows:

$$
\frac{\partial \mathcal{L}}{\partial W} = \sum_i \left(f(x_i) - y_i\right) \cdot f(x_i) \cdot \left(1 - f(x_i)\right) \cdot X_i
$$

Here, $f(x_i)$ represents the sigmoid function applied to the input $x_i$ associated with data point $i$.

#### 2.2 Derivative of Loss with Respect to $B$

Similarly, the **partial derivative of the loss function with respect to bias ($\frac{\partial \mathcal{L}}{\partial B}$)** is derived as:

$$
\frac{\partial \mathcal{L}}{\partial B} = \sum_i \left(f(x_i) - y_i\right) \cdot f(x_i) \cdot \left(1 - f(x_i)\right)
$$

The introduction of $X_i$ is omitted in this case, as it pertains to the bias term.

### 3. Algorithm Execution

The algorithmic execution involves several key steps:

1. **Initialization:**
   - Random initialization of weights ($W$) and bias ($B$).
   - Setting the learning rate ($\eta$) and maximum iterations.

2. **Gradient Computation:**
   - Iterating over all data points, computing the partial derivatives for $W$ and $B$ using the derived formulas.

3. **Parameter Update:**
   - Applying the gradient descent update rule to iteratively adjust the weights and bias.

### 4. Loss Surface Visualization

The lecture introduces the concept of visualizing the **loss function surface** in the $W-B$ plane. This visual aid illustrates the algorithm's movement along the surface, consistently reducing the loss.

### 5. Observations

The lecture emphasizes crucial observations:

- **Loss Reduction:**
  - Ensuring that at each iteration, the algorithm systematically decreases the loss.
  
- **Hyperparameter Impact:**
  - Acknowledging the influence of the learning rate ($\eta$) on convergence and potential overshooting.
  
- **Experimentation:**
  - Encouraging experimentation with diverse initializations and learning rates for a comprehensive understanding.

# Representation Power of Multi-Layer Networks

## Introduction

The representation power of a multi-layer network, particularly employing sigmoid neurons, is the focal point of this discussion. The objective is to establish a theorem analogous to the one developed for perceptrons, specifically emphasizing the network's capability to approximate any continuous function.

## Universal Approximation Theorem

The Universal Approximation Theorem posits that a multi-layer network with a single hidden layer possesses the capacity to approximate any continuous function with precision. This approximation is achieved by manipulating the weights and biases associated with the sigmoid neurons within the hidden layer.

## Tower Functions Illustration

To illustrate the approximation process, the concept of towers of functions is introduced. This entails deconstructing an arbitrary function into a summation of tower functions, wherein each tower is represented by sigmoid neurons. The amalgamation of these towers serves to approximate the original function.

## Tower Construction Process

The construction of towers involves the utilization of sigmoid neurons with exceptionally high weights, approaching infinity. This strategic choice mimics step functions. By subtracting these step functions, a tower-like structure is formed. Notably, the width and position of the tower are modulated by adjusting the biases of the sigmoid neurons.

## Tower Maker Neural Network

### Architecture

The lecture introduces a neural network architecture termed the "Tower Maker." This architecture comprises two sigmoid neurons characterized by high weights. The subtraction of their outputs yields a function resembling a tower.

### Sigmoid Neuron Configuration

The sigmoid neurons within the Tower Maker are configured with exceedingly high weights, akin to infinity. This configuration transforms the sigmoid functions into step functions, pivotal in constructing tower-like shapes.

### Bias Adjustment

Control over the width and position of the tower is exercised through the manipulation of biases associated with the sigmoid neurons. Adjusting these biases ensures the customization of the tower function according to specific requirements.

## Linear Function Integration

An additional layer is incorporated into the Tower Maker architecture to integrate linear functions. This augmentation enhances the network's ability to generate tower functions based on the input parameters.

## Network Adjustment for Precision

The lecture underscores the correlation between the desired precision (represented by epsilon) and the network's complexity. As the precision requirement increases, a more intricate network with an augmented number of neurons in the hidden layer becomes imperative. However, it is acknowledged that practical implementation may encounter challenges as the network's size expands.

## Single Input Function

Consider a function with a single input ($x$) plotted on the x-axis and corresponding output ($y$) on the y-axis. This introductory scenario involves the use of a sigmoid neuron function, denoted by:

$$f(x) = \frac{1}{1 + e^{-(wx + b)}}$$

where:

- $w$ represents the weight associated with the input.
- $b$ is the bias term.
- The sigmoid function smoothly transitions between 0 and 1.

## Two Input Function

Expanding the scope to a two-input function, let's consider an example related to oil mining, where salinity ($x_1$) and pressure ($x_2$) serve as inputs. The challenge is to establish a decision boundary separating points indicating the presence (orange) and absence (blue) of oil.

A linear decision boundary proves inadequate, prompting the need for a more complex function.

## Building a Tower in 2D

To construct a tower-like structure, two sigmoid neurons are introduced, each handling one input ($x_1$ and $x_2$). The sigmoid function takes the form:

$$f(x) = \frac{1}{1 + e^{-(w_ix_i + b)}}$$

Here, $i$ denotes the input index (1 or 2), $w_i$ is the associated weight, and $b$ is the bias term. Adjusting weights ($w_1$ and $w_2$) results in step functions, dictating the slope of the tower in different directions.

Combining these sigmoid neurons produces an open tower structure in one direction.

## Closing the Tower

To enclose the tower from all sides, two additional sigmoid neurons (h13 and h14) are introduced. These neurons, with specific weight configurations, contribute to the formation of walls in different directions. Subtracting the outputs of these sigmoid neurons results in a structure with walls on all four sides but an open top.

## Thresholding to Get a Closed Tower

To address the open top issue, thresholding is introduced. A sigmoid function with a switch-over point at 1 is applied to the structure's output. This process retains only the portion of the structure above level 1, effectively closing the tower.

## Extending to Higher Dimensions

Generalizing this approach to n-dimensional inputs, the methodology remains consistent. For a single input ($x$), two neurons suffice; for two inputs ($x_1$ and $x_2$), four neurons are necessary. The number of neurons in the middle layer increases with higher dimensions, extending the method to handle arbitrary functions.

## Universal Approximation Theorem

This construction aligns with the Universal Approximation Theorem, asserting that a neural network, given a sufficient number of neurons, can approximate any arbitrary function to a desired precision.

## Implications for Deep Learning

This methodology underscores the flexibility of deep neural networks in approximating complex functions encountered in real-world applications. The ability to systematically construct networks capable of representing intricate relationships contributes to the effectiveness of deep learning models.

# Conclusion

In this week's deep learning lectures, we delved into fundamental concepts, challenges, and advancements in the field. We explored the intricacies of boolean functions and their linear separability, shedding light on the limitations perceptrons face when dealing with complex functions like XOR. The introduction of multi-layer perceptrons (MLPs) provided a solution, extending the capability to handle non-linearly separable data.

The transition from perceptrons to sigmoid neurons marked a crucial shift, addressing the binary output limitation and introducing real-valued inputs and outputs. We explored the importance of the sigmoid function's differentiability in machine learning algorithms, particularly in the context of neural network training.

Supervised machine learning setups, learning parameters through trial-and-error and Taylor series approximation, and the mathematical foundations of gradient descent were thoroughly discussed. The optimization process for sigmoid neurons through gradient descent provided insights into updating weights and biases iteratively, aiming to minimize the loss function.

The representation power of multi-layer networks, illustrated through the Universal Approximation Theorem, showcased the ability of neural networks to approximate any continuous function. The Tower Maker architecture exemplified the construction of towers of functions using sigmoid neurons, highlighting the flexibility and power of deep neural networks.

The week concluded with an exploration of the Universal Approximation Theorem's implications for deep learning, emphasizing the adaptability of neural networks in approximating complex functions encountered in real-world applications.

## Points to Remember

1. **Boolean Functions and Linear Separability:**
   - Perceptrons face challenges with non-linearly separable boolean functions.
   - Multi-layer perceptrons (MLPs) extend capabilities for handling complex functions.

2. **Sigmoid Neurons and Differentiability:**
   - Sigmoid neurons introduce smoothness in decision-making.
   - Differentiability is crucial for optimization in neural network training.

3. **Gradient Descent: Mathematical Foundation:**
   - Taylor series expansion facilitates linear approximation in gradient descent.
   - Decision criteria for parameter updates involve linear approximation conditions.

4. **Tower Maker and Universal Approximation Theorem:**
   - The Universal Approximation Theorem states that a single hidden layer in a neural network can approximate any continuous function.
   - Tower Maker architecture showcases the construction of towers using sigmoid neurons.

5. **Deep Learning Flexibility:**
   - Deep neural networks are flexible in approximating complex functions.
   - The Tower Maker architecture demonstrates the power of neural networks in constructing intricate representations.

This week's exploration laid the groundwork for understanding the core principles and capabilities of neural networks, setting the stage for further exploration into advanced topics in deep learning.