# Decoder Stack

The decoder is a stack of $N = 6$ layers. Each layer consists of three sublayers:

## Layer Structure

1. **Masked Multi-Head (Self) Attention**
2. **Multi-Head (Cross) Attention**
3. **Feed Forward Network**

Each of the six layers (Layer 1 through Layer 6) follows this structure.  The provided examples (Layer 1, Layer 2, Layer 6) are redundant and only serve to illustrate the consistent layer composition.

## Input Example

### Input Tokens

The example provides a list of input tokens:  "Mitesh Khapraaa", "transformar", "padathchai", "rasithen".

A diagram, labeled "Decoder Stack Diagram", is included but its content is not described because the base64 encoded image data is missing.  [Missing Image: Decoder Stack Diagram]


# Teacher Forcing

Teacher forcing is an alternative training approach for sequence prediction models like decoders in LLMs.  Instead of feeding the decoder's previous prediction as input for the next prediction, teacher forcing feeds the *actual* target word (ground truth) at each step.

This addresses the issue of error accumulation: if the first prediction is incorrect, subsequent predictions are likely to be wrong as well when using only previous predictions. Teacher forcing mitigates this by guiding the model with the correct sequence, improving training speed and potentially leading to better performance.

A diagram illustrating teacher forcing shows the decoder receiving the ground truth target sentence word by word, generating predictions based on this input.  [Missing Image: Teacher Forcing Diagram]

**Example:**

The target sentence is "Enjoyed the sunshine last night".  The decoder's predictions are compared against this target.

| Decoder Step | Input (Target Word) | Prediction |
|---|---|---|
| D1 | enjoyed | enjoyed |
| D2 | the sunshine | the sunshine |
| D3 | last night | last night |
| D4 | enjoyed  | enjoyed |
| D5 | the film | the film |
| D6 | transforme | transforme |


The initial input is 'I' and the target sentence is "Enjoyed the sunshine last night".  The decoder makes predictions for each word in the target sentence, step-by-step, using the ground truth as input for each step.


# Decoder Layer Self-Attention Mechanism

The decoder layers, like the encoder, utilize self-attention.  Query ($Q$), key ($K$), and value ($V$) vectors are computed by multiplying target sentence word embeddings ($h_1, h_2, \ldots, h_T$) with transformation matrices ($W_Q, W_K, W_V$ respectively).

## Teacher Forcing Implementation via Masking

A crucial difference in the decoder's self-attention mechanism is the inclusion of a **mask** to implement teacher forcing during training.  This mask prevents the decoder from attending to future tokens in the target sequence. During inference, this mask is not used, and the decoder operates as an autoregressive model.

## Diagram: Decoder Self-Attention Mechanism

[Diagram Placeholder: This diagram shows the computation of the self-attention mechanism within a decoder layer, illustrating the steps: matrix multiplication of $Q^T$ and $K$, scaling by $\frac{1}{\sqrt{d_k}}$, application of the mask, softmax to obtain attention weights, and final matrix multiplication to compute the weighted sum of value vectors.]


## Encoder Masking

Encoder blocks also employ masking in their attention sublayers to mask padded tokens in sequences of length $T$.


# Masked Multi-Head Self-Attention: Mask Creation and Placement

## Mask Creation and Incorporation

The notes describe how to create and incorporate a mask into the self-attention mechanism.  The mask is used to prevent the decoder from attending to future tokens during training (teacher forcing).

### Equations for Q, K, V

The query ($Q$), key ($K$), and value ($V$) matrices are computed as follows:

$Q_1 = W_{Q_1} H$
$K_1 = W_{K_1} H$
$V_1 = W_{V_1} H$

where $H$ is the matrix of word embeddings, and $W_{Q_1}$, $W_{K_1}$, and $W_{V_1}$ are the respective transformation matrices.

### Masking Mechanism

The masking mechanism sets the attention weights ($\alpha_{ij}$) to zero for masked value vectors ($v_j$).  This is accomplished by adding a mask matrix ($M$) to the attention matrix ($A$) before applying the softmax function:

$Z = \text{softmax}(A + M) V^T$

where $A = Q_1^T K_1$.  The resulting matrix $Z$ represents the context vector after applying the mask and attention weights.

### Diagram Description

[Diagram Placeholder:  The diagram likely shows a visual representation of the masked multi-head self-attention mechanism, illustrating the calculation of Q, K, V matrices, the creation of the attention matrix A, the addition of the mask M, the softmax function, and the final weighted sum to obtain Z.]


### Example Input Tokens

The example provides a list of input tokens: "<GO>", "Naan", "transformers", "padathai", "rosiththen".  These tokens are presumably used as input to the masked multi-head self-attention layer.


# Masking in Matrix Representation for Decoder Self-Attention

Masking is implemented by inserting negative infinity ($-\infty$) at the relevant positions within the attention matrix.

## Matrix Representation of Masked Attention

A masked attention matrix $T$ is shown below, where $q_i \cdot k_j$ represents the dot product of the query vector $q_i$ and the key vector $k_j$.  The negative infinities prevent the model from attending to future tokens.

```
T = [
    [q1•k1, -∞, -∞, -∞, -∞],
    [q2•k1, q2•k2, -∞, -∞, -∞],
    [q3•k1, q3•k2, q3•k3, -∞, -∞],
    [q4•k1, q4•k2, q4•k3, q4•k4, -∞],
    [q5•k1, q5•k2, q5•k3, q5•k4, q5•k5]
]
```

## Triangular Mask Matrix

This creates a lower triangular matrix where the upper triangle is filled with negative infinity.  A separate mask matrix $M$ can represent this:

```
M = [
    [0, -∞, -∞, -∞, -∞],
    [0, 0, -∞, -∞, -∞],
    [0, 0, 0, -∞, -∞],
    [0, 0, 0, 0, -∞],
    [0, 0, 0, 0, 0]
]
```

Adding this mask matrix $M$ to the attention matrix $A$ ($A = Q^T K$) before applying the softmax function ensures that the attention weights for future tokens are effectively zero.  This is because $\text{softmax}(-\infty) \approx 0$.


# Multi-Head Cross Attention

## Cross-Attention Mechanism

Multi-head cross-attention in the decoder uses vectors from two sources:

1.  $\{s_1, s_2, \ldots, s_T\}$: Output vectors from the decoder's self-attention layer.
2.  $\{e_1, e_2, \ldots, e_T\}$: Vectors from the top layer of the encoder stack.  This is shared across all decoder layers.

Query ($Q$), key ($K$), and value ($V$) matrices are created using linear transformations:

$Q_2 = W_{Q_2} S$
$K_2 = W_{K_2} E$
$V_2 = W_{V_2} E$

where $S$ is the matrix of vectors from the self-attention layer and $E$ is the matrix of vectors from the encoder's top layer.  $W_{Q_2}$, $W_{K_2}$, and $W_{V_2}$ are the corresponding transformation matrices.

The cross-attention mechanism is then calculated as:

$Z = \text{softmax}(Q_2^T K_2)V_2^T$

The resulting vectors are concatenated across multiple attention heads, then passed through a feed-forward neural network to produce the output vectors $O$.


# Decoder Layer Parameter Count

## Masked Multi-Head Self-Attention Layer: Approximately 1 Million Parameters

## Multi-Head Cross-Attention Layer: Approximately 1 Million Parameters

## Feed Forward Network (FFN) Layer: Approximately 2 Million Parameters

The FFN parameter count is calculated as: $2 \times (512 \times 2048) + 2048 + 512 = 2,097,152 \approx 2 \times 10^6$.  This calculation assumes an FFN with two linear layers, where the first has 512 input units and 2048 hidden units, the second has 2048 input units and 512 output units.  The additional terms (2048 and 512) likely represent biases for the two layers.

## Total Decoder Layer Parameters: Approximately 4 Million Parameters

This is the sum of the parameters in the Masked Multi-Head Self-Attention, Multi-Head Cross-Attention, and FFN layers.


# Positional Encoding

## Methods for Filling Positional Vectors ($p_0$)

The notes explore different approaches to filling the elements of a positional vector $p_0$:

1. **Constant Vector:**  All elements are a constant value $j$ for $p_j$.
2. **One-Hot Encoding:**  Using one-hot encoding to represent the position $j$, where $j = 0, 1, \ldots, T$.
3. **Learned Embeddings:** Learning embeddings for all possible positions.

The notes state that the third option is unsuitable for sentences of dynamic length.

## Diagram Placeholder

[Diagram Placeholder: This diagram likely illustrates the different positional encoding methods, possibly showing the vector representations for each method.]

## Vector Dimensions and Notation

The notes show a vector $h_0 \in \mathbb{R}^{512}$, implying that the word embedding has 512 dimensions.  The symbol $\oplus$ suggests a vector addition operation.  The index $i$ ranges from 0 to 511, corresponding to the 512 dimensions.  The vector $p_0$ represents the positional encoding for the first position.


# Sinusoidal Positional Encoding

A hypothesis is proposed: embedding a unique pattern of features for each position $j$ allows the model to learn to attend based on relative position.

## Sinusoidal Encoding Function

The features are generated using the following function:

$ PE_{(j, i)} = \begin{cases}
\sin \left( \frac{j}{10000^{\frac{2i}{d_{model}}}} \right) & \text{if } i \text{ is even} \\
\cos \left( \frac{j}{10000^{\frac{2i-1}{d_{model}}}} \right) & \text{if } i \text{ is odd}
\end{cases} $

where $d_{model} = 512$.

For a fixed position $j$, the value for $i$ is sampled from $\sin()$ if $i$ is even or $\cos()$ if $i$ is odd.

## Visualization

The function $PE_{(j, i)}$ is evaluated for $j = 0, 1, \ldots, 8$ and $i = 0, 1, \ldots, 63$, and the resulting matrix is visualized as a heatmap.  [Missing Heatmap: This heatmap would show the sinusoidal positional encodings for the specified range of j and i values.]


# Module 16.4: Training the Transformer

## Transformer Architecture (Summary)

This section summarizes the architecture of transformers,  briefly listing components already detailed in the previous notes: Self-Attention Mechanism, Multi-Head Attention, Position Encoding, Feed-Forward Neural Networks, Encoder and Decoder Structure.  No new information is provided.


## Training Procedure

### Data Preparation

This step is mentioned but not detailed.

### Loss Function

The loss function used is specified as `nn.CrossEntropyLoss()` in the example code.

### Optimization Algorithms

Gradient Descent and the Adam Optimizer are mentioned as optimization algorithms used during training.

### Training Loop

The training loop is described using a Python code snippet:

```python
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

This code shows the basic structure of a training loop, including forward pass, backward pass, and gradient updates.

## Optimization Techniques

Learning Rate Scheduling, Regularization Techniques, Batch Normalization, and Dropout are listed as optimization techniques. No details are provided.


## Performance Metrics

Accuracy, Precision, Recall, F1-Score, BLEU Score (for NLP tasks), and Perplexity are listed as performance metrics.


## Example Code for Training a Transformer

A basic Python code example is given showing the structure for training a transformer model using PyTorch.  It includes model definition, loss function, optimizer, and a training loop.  The actual implementation of the transformer layers is left unspecified.  The code is partially redundant as it only shows the training loop structure which was already conceptually outlined in the previous notes.


## Applications

Natural Language Processing (NLP) applications such as Machine Translation, Text Summarization, and Sentiment Analysis are mentioned, along with Computer Vision applications like Image Classification and Object Detection.  This information is not new.



# Layer Normalization

Layer normalization is a normalization technique applied to the outputs of hidden units within a single layer.  Unlike batch normalization, it computes the mean and variance across the hidden units of a single layer, independent of the batch size. This allows for training with batch size 1, useful for recurrent neural networks (RNNs).

## Layer Normalization Formulas

The formulas for layer normalization are as follows:

**1. Mean ($μ_l$)**:

$μ_l = \frac{1}{H} \sum_{i=1}^{H} x_i$

where $H$ is the number of hidden units in the layer, and $x_i$ is the output of the $i$-th hidden unit.


**2. Variance ($σ_l$)**:

$σ_l = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (x_i - μ_l)^2}$


**3. Normalized Output ($\hat{x_i}$)**:

$\hat{x_i} = \frac{x_i - μ_l}{\sqrt{σ_l^2 + ε}}$

where $ε$ is a small constant added for numerical stability.


**4. Scaled and Shifted Output ($\hat{y_i}$)**:

$\hat{y_i} = γ\hat{x_i} + β$

where $γ$ and $β$ are learned scaling and shifting parameters, respectively.


**[Diagram Placeholder]:**  A cube representing the layer normalization process.  The cube likely represents the activation values of a layer, showing how the mean and variance are calculated across the hidden units.  The normalized and scaled outputs are then produced.


