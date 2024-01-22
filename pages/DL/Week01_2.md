# Motivation from Biological Neuron

## Introduction

Artificial neurons, the foundational units in artificial neural networks, find their roots in biological neurons, a term coined in the 1890s to describe the brain's processing units.

## Biological Neurons

### Components
- **Dendrite:** Functions as a signal receiver from other neurons.
- **Synapse:** The connection point between neurons.
- **Soma:** The central processing unit for information.
- **Axon:** Transmits processed information to other neurons.

### Illustration
In a simplified depiction, sense organs interact with the external environment, and neurons process this information, potentially resulting in physical responses, such as laughter.

## Neural Network Architecture

- **Layered Structure:** Neurons are organized into layers.
- **Interconnected Network:** The human brain comprises approximately 100 billion neurons.
- **Division of Work:** Neurons may specialize in processing specific information types.
- **Example:** Neurons responding to visual, auditory, or textual stimuli.

## Multi-Layer Perceptrons (MLPs)

### Definition
Neural networks with multiple layers.

### Information Processing
Initial neurons interact with sensory organs, and subsequent layers perform increasingly intricate processing.

### Demonstration
Using a cartoon illustration: Neurons in the visual cortex detect edges, form features, and recognize objects.

### Layer Functions
1. **Layer 1:** Detects edges and corners.
2. **Subsequent Layers:** Organize information into features and recognize complex objects.

### Abstraction
Each layer processes more abstract representations of the input.

### Information Flow
Input traverses through layers, resulting in a physical response.

# McCulloch-Pitts Neuron and Thresholding Logic

## Introduction

- **Objective:** Comprehend the McCulloch-Pitts neuron, a simplified computational model inspired by biological neurons.
- **Historical Context:** Proposed in 1943 by McCulloch (neuroscientist) and Pitts (logician).
- **Purpose:** Emulate the brain's complex processing for decision-making.

## Neuron Structure

- **Components:** Divided into two parts - **g** and **f**.
- **g (Aggregation):** Aggregates binary inputs via a simple summation process.
- **f (Decision):** Makes a binary decision based on the aggregation.
- **Excitatory and Inhibitory Inputs:** Inputs can be either excitatory (positive) or inhibitory (negative).

## Functionality

- **Aggregation Function g(x):**
  - Represents the sum of all inputs using the formula $g(x) = \sum_{i=1}^{n} x_i$, where $x_i$ is a binary input (0 or 1).
- **Decision Function f(g(x)):**
  - Utilizes a threshold parameter $\theta$ to determine firing.
  - Decision is $f(g(x)) = \begin{cases} 1 & \text{if } g(x) \geq \theta \\ 0 & \text{otherwise} \end{cases}$.
\newpage

## Boolean Function Implementation

- **Examples:**
  - Implemented using McCulloch-Pitts neuron for boolean functions like AND, OR, NOR, and NOT.
  - Excitatory and inhibitory inputs utilized based on boolean function logic.

## Geometric Interpretation

- **In 2D:**
  - Draws a line to separate input space into two halves.
- **In 3D:**
  - Uses a plane for separation.
- **For n Inputs:**
  - Utilizes a hyperplane for linear separation.

## Linear Separability

- **Definition:** Boolean functions representable by a single McCulloch-Pitts neuron are linearly separable.
- **Implication:** Implies the existence of a plane (or hyperplane) separating points with output 0 and 1.

# Perceptrons and Boolean Functions

## Introduction

Perceptrons, introduced by Frank Rosenblatt circa 1958, extend the concept of McCulloch-Pitts neurons with non-Boolean inputs, input weights, and a learning algorithm for weight adjustment.

## Perceptron Model

### Mathematical Representation

The perceptron is represented as $y = 1$ if $\sum_{i=1}^{n} w_i x_i \geq \text{threshold}$; otherwise, $y = 0$.

#### Notable Differences

1. Inputs can be real, not just Boolean.
2. Introduction of weights, denoted by $w_i$, indicating input importance.
3. Learning algorithm to adapt weights based on data.

### Neater Formulation

The equation is rearranged for simplicity: $\sum_{i=0}^{n} w_i x_i \geq 0$, where $x_0 = 1$ and $w_0 = -\text{threshold}$.

## Motivation for Boolean Functions

Boolean functions provide a foundation for understanding perceptrons. For instance, predicting movie preferences using Boolean inputs such as actor, director, and genre.

## Importance of Weights

Weights signify the importance of specific inputs in decision-making. Learning from data helps adjust weights, reflecting user preferences. For example, assigning a high weight to the director may heavily influence the decision to watch a movie.

## Bias ($w_0$)

$w_0$ acts as a bias or prior, influencing decision-making. It represents the initial bias or prejudice in decision-making. Adjusting $w_0$ alters the decision threshold, accommodating user preferences.

## Implementing Boolean Functions

Perceptrons can implement Boolean functions with linear decision boundaries. For instance, implementing the OR function with a perceptron involves a geometric interpretation where a line separates positive and negative regions based on inputs.

## Errors and Adjustments

Errors arise when the decision boundary misclassifies inputs. The learning algorithm adjusts weights iteratively to minimize errors and enhance accuracy. It's an iterative process where weights are modified until the desired decision boundary is achieved.

# Errors and Error Surfaces

## Introduction

This section delves into errors within the context of perceptrons and introduces error surfaces as a recurring theme in the course, with a focus on understanding errors related to linear separability.

## Perceptron for AND Function

Consideration of the AND function showcases an output of 1 for a specific input (green) and 0 for others (red). The decision is based on $w_0 + w_1x_1 + w_2x_2 \geq 0$, with $w_0$ fixed at -1. Exploration of the impact of $w_1$ and $w_2$ on the decision boundary is undertaken.

## Errors and Decision Boundaries

Demonstration of errors occurs with specific $w_1$ and $w_2$ values, showcasing misclassified points due to incorrect decision boundaries. Variability in errors is noted based on different weight values.

## Error Function

Viewing error as a function of $w_1$ and $w_2$ is introduced. The concept of error surfaces is brought in, where error is plotted against $w_1$ and $w_2$ values, each region on the surface corresponding to a distinct error level.

## Visualizing the Error Surface

The error surface is plotted for $w_1$ and $w_2$ values in the range -4 to +4. Each region on the surface corresponds to a distinct error level, highlighting the utility of visualizations in comprehending perceptron behavior.

## Perceptron Learning Algorithm

Exploration of the necessity for an algorithmic approach to finding optimal $w_1$ and $w_2$ values is undertaken. Limitations in visual inspection, especially in higher dimensions, are acknowledged. A teaser for the upcoming module on the perceptron learning algorithm is provided as a solution for finding suitable weight values algorithmically.

# Perceptron Learning Algorithm

## Overview

This module focuses on the Perceptron Learning Algorithm, building upon the perceptron's concept and introducing a method to iteratively adjust weights for accurate binary classification.

## Motivation

The perceptron, initially designed for boolean functions, finds practical application in real-world scenarios. Consider a movie recommendation system based on past preferences, where features include both boolean and real-valued inputs. The goal is to learn weights that enable accurate predictions for new inputs.

## Algorithm

### Notations
- $p$: Inputs with label 1 (positive points)
- $n$: Inputs with label 0 (negative points)

### Convergence
Convergence is achieved when all positive points satisfy $\sum w_i x_i > 0$ and all negative points satisfy $\sum w_i x_i < 0$.

### Steps
1. **Initialization**: Randomly initialize weights $w$.
2. **Iterative Update**:
   - While not converged:
      - Pick a random point $x$ from $p \cup n$.
      - If $x$ is in $p$ and $w^T x < 0$, update $w = w + x$.
      - If $x$ is in $n$ and $w^T x \geq 0$, update $w = w - x$.

## Geometric Interpretation

Understanding the geometric relationship involves recognizing that the angle between $w$ and a point on the decision boundary is 90 degrees. Positive points' angles should be acute (< 90 degrees), and negative points' angles should be obtuse (> 90 degrees). Iteratively adjusting $w$ aligns it better with correctly classified points.

# Perceptron Convergence Proof

## Introduction

The objective of this lecture is to present a formal proof establishing the convergence of the perceptron learning algorithm. The primary focus is to rigorously determine whether the algorithm exhibits convergence or continues weight updates indefinitely.
\newpage

## Definitions

1. **Absolutely Linearly Separable Sets**
   - Consider two sets, $P$ and $N$, in an $n$-dimensional space. They are deemed absolutely linearly separable if there exist $n + 1$ real numbers $w_0$ to $w_n$ such that the following conditions hold:
    $$
    w_0x_0 + w_1x_1 + \ldots + w_nx_n \geq 0 \quad \text{for every } \mathbf{x} \in P
    $$
    $$
    w_0x_0 + w_1x_1 + \ldots + w_nx_n < 0 \quad \text{for every } \mathbf{x} \in N
    $$

2. **Perceptron Learning Algorithm Convergence Theorem**
   - If sets $P$ and $N$ are finite and linearly separable, the perceptron learning algorithm will update the weight vector a finite number of times. This implies that after a finite number of steps, the algorithm will find a weight vector $\mathbf{w}$ capable of separating sets $P$ and $N$.

## Proof

### Setup
Define $P'$ as the union of $P$ and the negation of $N$. Normalize all inputs for convenience.

### Assumptions and Definitions
Assume the existence of a normalized solution vector $\mathbf{w^*}$. Define the minimum dot product, $\delta$, as the minimum value obtained by dot products between $\mathbf{w^*}$ and points in $P'$.

### Perceptron Learning Algorithm
The perceptron learning algorithm can be expressed as follows:

1. **Initialization:**
   - Initialize weight vector $\mathbf{w}$ randomly.

2. **Iteration:**
   - At each iteration, randomly select a point $\mathbf{p}$ from $P'$.
   - If the condition $\mathbf{w}^T\mathbf{p} \geq 0$ is not satisfied, update $\mathbf{w}$ by $\mathbf{w} = \mathbf{w} + \mathbf{p}$.

### Normalization and Definitions
Normalize all inputs, ensuring the norm of $\mathbf{p}$ is 1. Define the numerator of $\cos \beta$ as the dot product between $\mathbf{w^*}$ and the updated weight vector at each iteration.

### Numerator Analysis
Show that the numerator is greater than or equal to $\delta$ for each iteration.

For a randomly selected $\mathbf{p}$, if $\mathbf{w}^T\mathbf{p} < 0$ and an update is performed, the numerator is:

$$
\mathbf{w^*} \cdot (\mathbf{w} + \mathbf{p}) \geq \delta
$$

### Denominator Analysis
Expand the denominator, the square of the norm of the updated weight vector:

$$
\|\mathbf{w} + \mathbf{p}\|^2 = \|\mathbf{w}\|^2 + 2\mathbf{w}^T\mathbf{p} + \|\mathbf{p}\|^2
$$

Show that the denominator is less than or equal to a value involving $k$, the number of updates made:

$$
\|\mathbf{w} + \mathbf{p}\|^2 \leq \|\mathbf{w^*}\|^2 + k
$$

### Combining Numerator and Denominator
Use the definition of $\cos \beta$ to conclude that $\cos \beta$ is greater than or equal to a certain quantity involving the square root of $k$:

$$
\cos \beta \geq \frac{\delta}{\sqrt{k}}
$$

# Conclusion

In this week's lectures on deep learning, we explored fundamental concepts related to artificial neural networks, focusing on the motivation drawn from biological neurons, the structure of McCulloch-Pitts neurons, and the evolution into perceptrons. We delved into the mathematical representations, functionalities, and the learning algorithm associated with perceptrons. Additionally, we discussed errors and error surfaces, highlighting their significance in understanding the behavior of perceptrons.

The exploration continued with an in-depth examination of the Perceptron Learning Algorithm, emphasizing its application in real-world scenarios. The algorithm's convergence was rigorously presented, providing a formal proof for its finite number of weight updates when dealing with linearly separable data.

# Points to Remember

1. **Biological Neurons:** Understanding the components of biological neurons, including dendrites, synapses, soma, and axons, served as the foundation for artificial neural networks.

2. **McCulloch-Pitts Neuron:** The simplified computational model, inspired by biological neurons, introduced the aggregation and decision functions, showcasing its application in implementing boolean functions.

3. **Perceptrons:** Extending the McCulloch-Pitts model, perceptrons introduced real-valued inputs, weights, and a learning algorithm for weight adjustments, emphasizing their significance in decision-making.

4. **Perceptron Learning Algorithm:** The iterative algorithm for adjusting weights in perceptrons was presented, highlighting its application in scenarios with boolean and real-valued inputs.

5. **Convergence Proof:** A formal proof established the convergence of the Perceptron Learning Algorithm for linearly separable data, emphasizing its practical applicability in real-world scenarios.

6. **Geometric Interpretation:** Recognizing the geometric relationship between the weight vector and decision boundaries provided insights into how the algorithm aligns with correctly classified points.

7. **Error Surfaces:** Visualizing error surfaces proved essential in comprehending perceptron behavior, showcasing the impact of weight adjustments on error levels.

8. **Practical Applications:** Motivation for the algorithm was drawn from real-world applications, such as movie recommendation systems, emphasizing the practicality and relevance of the discussed concepts.

9. **Necessity of Learning Algorithm:** The iterative nature of weight adjustments was emphasized, recognizing the limitations of visual inspection, especially in higher dimensions.

10. **Implications of Convergence Proof:** The convergence proof provided assurance that, when dealing with linearly separable data, the perceptron learning algorithm will converge after a finite number of updates.
