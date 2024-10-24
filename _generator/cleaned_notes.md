---
title: ## Transformer Decoder Explained 

---

# Decoder Stack

The decoder is a stack of $N = 6$ layers. Each layer consists of three sublayers: Masked Multi-Head (Self) Attention, Multi-Head (Cross) Attention, and a Feed Forward Network.

## Layer Structure

Each of the six layers in the decoder stack has the same structure:

1. Masked Multi-Head (Self) Attention
2. Multi-Head (Cross) Attention
3. Feed Forward Network

## Input Example

The provided input tokens are: "Mitesh Khapraaa", "transformar", "padathchai", and "rasithen".


# Teacher Forcing

Teacher forcing is a training technique for sequence-to-sequence models, like those used in machine translation.  Instead of using the model's previous prediction as input for the next prediction, teacher forcing feeds the *ground truth* (the actual target sequence) as input at each step.

This addresses the problem of error accumulation: if an early prediction is incorrect, subsequent predictions will likely also be incorrect due to the conditional nature of the model. Teacher forcing mitigates this by guiding the model towards the correct output at each step during training. While the model still needs to learn to handle errors during inference (when it only receives its own predictions as input), teacher forcing speeds up training by preventing the cascading effect of early errors.

Example:

**Initial Input:** I
**Target Sentence:** Enjoyed the sunshine last night
**Predictions:**
- D1: enjoyed
- D2: the sunshine
- D3: last night
- (Further examples might exist for different sentences, such as "Enjoyed the film transform")


# Masked (Self) Attention

## Components

The masked self-attention mechanism uses the following components:

1. **MatMul:** Matrix multiplication.
2. **Softmax:** Applies the softmax function to the output of the matrix multiplication.
3. **Scale: $\frac{1}{\sqrt{d_k}}$:** Scales down the output of the softmax by the square root of the dimension of the key vectors ($d_k$). This helps to stabilize training.
4. **MatMul: $Q^T K$**: Matrix multiplication of the transpose of the query matrix ($Q^T$) and the key matrix ($K$). This computes the attention weights.


## Inputs

The inputs to the masked self-attention mechanism are:

- $Q$: Query matrix.
- $K$: Key matrix.
- $V$: Value matrix.


## Explanation

In the decoder layers, the query, key, and value vectors ($q, k, v$) are computed by multiplying the target sentence's word embeddings ($h_1, h_2, \ldots, h_T$) with the transformation matrices ($W_Q, W_K, W_V$) respectively. This is similar to the self-attention mechanism in the encoder, but with one crucial difference: The details of this difference are not explicitly stated in the provided text.


# Decoder Layer Masking: Teacher Forcing and Autoregression

The decoder layers use the same computation of query, key, and value vectors ($q, k, v$) as the encoder, multiplying the target sentence's word embeddings ($h_1, h_2, \ldots, h_T$) with transformation matrices ($W_Q, W_K, W_V$).  However, a crucial difference is the addition of a **mask** to implement teacher forcing during training.  During training, this mask prevents the model from attending to future tokens in the target sentence.

During inference, teacher forcing is not used, and the decoder acts as an autoregressor, using its previous predictions as input for the next prediction.


# Encoder Masking

The encoder block also uses masking in its attention sublayer to mask padded tokens in sequences of length $T$.  This is done to ignore padded parts of the input sequence during the self-attention calculation.


# Masked Multi-Head Self Attention: Mask Creation and Incorporation

## Mask Creation and Placement

This section describes how to create and incorporate a mask into the masked multi-head self-attention mechanism.  The mask is used to prevent the model from attending to future tokens in a sequence.

### Equations for Query, Key, and Value Matrices

$Q_1 = W_{Q_1} H$
$K_1 = W_{K_1} H$
$V_1 = W_{V_1} H$

where $H$ is the matrix of word embeddings, and $W_{Q_1}$, $W_{K_1}$, and $W_{V_1}$ are the corresponding transformation matrices.

### Attention Weights and Masking

The attention weights ($A$) are computed as the matrix product of the transpose of the query matrix and the key matrix:

$Q_1^T K_1 = A$

Masking is applied by setting certain weights in $A$ to zero.  The mask matrix ($M$) is added to $A$ before the softmax function is applied.  The masked attention weights are then used to compute the weighted sum of the value vectors ($V^T$).

$Z = \text{softmax}(A + M) V^T$

where $Z$ is the output of the masked multi-head self-attention.

### Mask Matrix ($M$)

The mask matrix $M$ is a matrix of the same dimensions as $A$, where elements $α_{ij} = 0$ indicate that the value vector $v_j$ should be masked for a given query vector $q_i$.  This effectively prevents the model from attending to future tokens in a sequence.


# Multi-Head Cross Attention

## Cross-Attention Mechanism

Multi-head cross-attention in the decoder uses vectors from both the decoder's self-attention layer ($s_1, s_2, \ldots, s_T$) and the encoder's top layer ($e_1, e_2, \ldots, e_T$).  The encoder vectors are shared across all decoder layers.

Query, key, and value matrices are created using linear transformations:

$Q_2 = W_{Q_2} S$  (Queries from decoder self-attention output)
$K_2 = W_{K_2} E$  (Keys from encoder top layer)
$V_2 = W_{V_2} E$  (Values from encoder top layer)

where $S$ represents the matrix of vectors from the self-attention layer and $E$ represents the matrix of vectors from the encoder's top layer.  $W_{Q_2}$, $W_{K_2}$, and $W_{V_2}$ are the corresponding transformation matrices.


The attention weights are calculated as:

$Z = \text{softmax}(Q_2^T K_2)V_2^T$

The multi-head attention is computed using $Q_2$, $K_2$, and $V_2$, and the resulting vectors are concatenated.  These concatenated vectors are then fed into a feed-forward neural network to produce the final output vectors $O$.


# Decoder Layer Parameter Count

## Masked Multi-Head Attention Layer: ~1 Million Parameters

## Multi-Head Cross Attention Layer: ~1 Million Parameters

## Feed Forward Network (FFN) Layer: ~2 Million Parameters

### FFN Parameter Calculation:

The number of parameters in the FFN layer is calculated as:  $2 \times (512 \times 2048) + 2048 + 512 = 2,097,152 \approx 2 \text{ million}$.  This calculation assumes a two-layer FFN with hidden dimension 2048 and input/output dimension 512.


## Total Decoder Layer Parameters: ~4 Million

Each decoder layer contains approximately 4 million parameters, comprising the parameters from the masked multi-head attention, multi-head cross-attention, and the FFN layer.


# Positional Encoding in Transformers

Transformers lack the inherent positional information present in RNNs' hidden states.  Self-attention's permutation-invariant nature necessitates explicit positional encoding.

## Embedding Positional Information

The challenge is to incorporate positional information into the existing word embeddings ($h_j$) of size 512.

## Positional Encoding Method

The provided text describes a positional encoding method but lacks the specific details of how the positional information is calculated and added to the word embeddings.


# Sinusoidal Positional Encoding

The positional encoding method uses a sinusoidal function to embed positional information into word embeddings.

## Sinusoidal Encoding Function

The function generates features for each position $j$:

$PE_{(j, i)} = \begin{cases}
\sin \left( \frac{j}{10000^{\frac{2i}{d_{model}}}} \right) & \text{if } i \text{ is even} \\
\cos \left( \frac{j}{10000^{\frac{2i-1}{d_{model}}}} \right) & \text{if } i \text{ is odd}
\end{cases}$

where $d_{model} = 512$.  For a given position $j$, the value for dimension $i$ is determined by either the sine or cosine function depending on whether $i$ is even or odd.


# Layer Normalization

Layer normalization is a normalization technique applied at the $l^{th}$ layer of a neural network.  It computes the mean and variance across the hidden units within a single layer, independently of the batch size.  This allows for training with a batch size of 1, which can be useful for recurrent neural networks (RNNs).

## Formulas for Layer Normalization

### Mean ($\mu_l$)

$$ \mu_l = \frac{1}{H} \sum_{i=1}^{H} x_i $$

where $H$ is the number of hidden units in the layer, and $x_i$ is the output of the $i$-th hidden unit.

### Variance ($\sigma_l$)

$$ \sigma_l = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (x_i - \mu_l)^2} $$

### Normalized Output ($\hat{x}_i$)

$$ \hat{x}_i = \frac{x_i - \mu_l}{\sqrt{\sigma_l^2 + \epsilon}} $$

where $\epsilon$ is a small constant added for numerical stability.

### Scaled and Shifted Output ($\hat{y}_i$)

$$ \hat{y}_i = \gamma \hat{x}_i + \beta $$

where $\gamma$ and $\beta$ are learned parameters that allow the network to scale and shift the normalized output.


# Transformer Architecture: Input Embeddings and Positional Encoding

The input word embeddings are learned during training, not using pretrained models like Word2Vec.  This adds a set of weights to the model. Positional information is encoded and added to the input embeddings; this encoding function can also be parameterized.  The output of the top encoder layer is fed as input to all decoder layers for multi-head cross-attention.


# Encoder Layer Structure

The encoder layer consists of:

1. **Input Embedding:** Includes positional encoding.
2. **Add & Norm:**
3. **Multi-Head Attention:** Uses keys (K), values (V), and queries (Q).
4. **Feed Forward:** A feed-forward neural network.
5. **Add & Norm:**


# Decoder Layer Structure

The decoder layer consists of:

1. **Output Embedding:** Includes positional encoding.
2. **Masked Multi-Head Attention:** Masks future tokens.
3. **Add & Norm:**
4. **Multi-Head Attention:** Cross-attention with the encoder output.
5. **Feed Forward:** A feed-forward neural network.
6. **Add & Norm:**


# Decoder Component: LuCUS

The decoder uses a component called "LuCUS".  The exact meaning and function of LuCUS requires further clarification.
