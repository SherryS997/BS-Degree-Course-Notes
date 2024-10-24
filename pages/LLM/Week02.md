---
title: Transformer Decoder Explained 
---

## Decoder Stack

The decoder is a stack of $N = 6$ identical layers.  Each layer is composed of three sublayers:

1. **Masked Multi-Head (Self) Attention:** This sublayer performs self-attention on the decoder's input, but masks future tokens to prevent the model from "cheating" during training.  This ensures that the prediction for a given token depends only on the preceding tokens.

2. **Multi-Head (Cross) Attention:** This sublayer performs attention over the output of the encoder.  It allows the decoder to focus on relevant parts of the input sequence when generating the output sequence.  The queries come from the decoder's previous sublayer (masked self-attention), while the keys and values come from the encoder's output.

3. **Feed Forward Network:** This is a position-wise feed-forward network applied to each position's output from the multi-head cross-attention sublayer. It consists of two linear transformations with a ReLU activation in between.


Each of these sublayers employs a residual connection around it, followed by layer normalization. This can be represented as:

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$
where $x$ is the input to the sublayer, and Sublayer represents the masked self-attention, cross-attention, or feed-forward network.

## Teacher Forcing

Teacher forcing mitigates error accumulation during decoder training.  In standard autoregressive decoding, each prediction is conditioned on the *previous prediction*.  A mistake early in the sequence can cascade, leading to subsequent errors and slower training.

Teacher forcing uses the *ground truth* (correct target sequence) as input at each timestep, alongside the previous prediction.  This provides a stronger learning signal, correcting errors immediately and facilitating faster convergence.

More formally, let $y = (y_1, y_2, ..., y_T)$ be the target sequence and $\hat{y} = (\hat{y}_1, \hat{y}_2, ..., \hat{y}_T)$ be the predicted sequence. In autoregressive decoding without teacher forcing:

$P(\hat{y}|x) = \prod_{t=1}^{T} P(\hat{y}_t|\hat{y}_{<t}, x)$

where $x$ is the input sequence.  With teacher forcing, the probability becomes:

$P(\hat{y}|x, y) = \prod_{t=1}^{T} P(\hat{y}_t|y_{<t}, x)$

This means the prediction at time $t$ is conditioned on the *actual* previous tokens $y_{<t}$ from the target sequence, instead of the predicted tokens $\hat{y}_{<t}$.

During inference, teacher forcing is disabled, and the model reverts to standard autoregressive decoding. The decoder acts as an auto-regressor, using its own previous predictions as input.  This ensures that during deployment, the model can generate sequences independently, without relying on ground truth.

## Masked (Self) Attention (Detailed)

Masked self-attention in the decoder operates similarly to standard self-attention but incorporates a mask to prevent the model from attending to future tokens.  This is crucial during training to ensure the model learns to predict the next token based only on the preceding context.

The process begins by calculating the query ($Q$), key ($K$), and value ($V$) matrices.  These are obtained by multiplying the input matrix $H$ (representing the embedded input tokens) with the learned weight matrices $W_Q$, $W_K$, and $W_V$ respectively:

$Q = H W_Q$
$K = H W_K$
$V = H W_V$


Next, the attention weights are calculated.  This involves a matrix multiplication of $Q$ and $K^T$, scaling by $1/\sqrt{d_k}$ (where $d_k$ is the dimension of the key vectors), and applying the softmax function.  The masking is applied *before* the softmax:


$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$


Where $M$ is the masking matrix. $M$ is an upper triangular matrix filled with negative infinity ($-\infty$).  Adding this to the attention scores before the softmax effectively zeros out the attention weights corresponding to future tokens.  This prevents information from future tokens from influencing the prediction of the current token.

The output of the softmax operation is a matrix of attention weights, where each row represents a token in the input sequence, and each column represents the attention given to other tokens (including itself).  Because of the mask, the attention weights for future tokens are zero.

Finally, these attention weights are multiplied with the value matrix $V$ to obtain the context vector for each token. This context vector is a weighted sum of the value vectors of all preceding tokens, where the weights are determined by the attention mechanism.

## Masking in Matrix Representation

Masking assigns zero weights ($\alpha_{ij} = 0$) to masked value vectors ($v_j$) in a sequence. This is achieved during the self-attention calculation by adding a mask matrix *M*  to the attention matrix *A* before applying the softmax function.

Given the query matrix *Q*, key matrix *K*, and value matrix *V*, the attention matrix *A* is calculated as:

$$ A = Q^T K $$

The mask matrix *M* is added to *A*:

$$ A' = A + M $$

Finally, the output *Z* is calculated using the softmax function:

$$ Z = \text{softmax}(A') V^T = \text{softmax}(A + M) V^T $$

The mask *M* is a triangular matrix.  The lower triangular portion (including the diagonal) consists of zeros, allowing attention to be computed between current and previous tokens.  The upper triangular portion contains negative infinity ($-\infty$).  During the softmax operation, the negative infinity values become effectively zero, preventing attention to subsequent (future) tokens in the sequence.  This mechanism is crucial for ensuring the decoder only attends to past tokens during training, mimicking the autoregressive behavior needed during inference.

For example, for a sequence of length 5, the mask *M* would be:

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty & -\infty \\
0 & 0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$
Adding *M* to *A* effectively zeros out the elements corresponding to future tokens in the attention matrix *A'*, ensuring the decoder doesn't "look ahead" during training.

## Masked Multi-Head Self Attention

The purpose of the masked multi-head self-attention mechanism within the decoder is to allow each position to attend to all preceding positions in the input sequence, including itself, while preventing attention to future positions.  This is crucial during training with teacher forcing to prevent the model from "cheating" by looking ahead at the target sequence.

The process begins by creating the query ($Q_1$), key ($K_1$), and value ($V_1$) matrices for the self-attention mechanism.  These are derived from the input matrix $H$ (which can be thought of as a sequence of word embeddings combined with positional encodings) and multiplied by learned weight matrices $W_{Q_1}$, $W_{K_1}$, and $W_{V_1}$, respectively.

$Q_1 = W_{Q_1} H$
$K_1 = W_{K_1} H$
$V_1 = W_{V_1} H$

The attention matrix $A$ is then calculated by performing a dot product between the transpose of the query matrix and the key matrix.

$A = Q_1^T K_1$

This attention matrix $A$ represents the pairwise similarities between all positions in the input sequence. However, since we want to prevent attention to future tokens, we apply a mask $M$ to this attention matrix.

The mask $M$ is an upper triangular matrix where the upper triangle (representing attention to future tokens) is filled with negative infinity ($-\infty$) and the lower triangle (representing attention to past and present tokens) is filled with zeros.

Adding the mask to the attention matrix effectively nullifies the attention weights for future tokens:

$$
A' = A + M
$$

Next, a softmax function is applied to the masked attention matrix $A'$ to obtain the attention weights matrix $Z$. These weights represent the normalized importance of each past token (including the current token) when generating the output for the current position.

$$
Z = \text{softmax}(A') = \text{softmax}(A + M)
$$


Finally, the output of the masked multi-head self-attention is calculated by a weighted sum of the value vectors $V_1$, where the weights are determined by the attention weights matrix $Z$.

$$
\text{Output} = Z V_1^T
$$

This output is then typically passed through a feed-forward network and a layer normalization step. The "multi-head" aspect involves repeating this process multiple times with different learned weight matrices ($W_{Q_1}$, $W_{K_1}$, $W_{V_1}$) for each "head," and concatenating the results before feeding them into the feed-forward network.  This allows the model to capture different aspects of the relationships between words in the sequence.

## Multi-Head Cross Attention

Multi-Head Cross Attention is a crucial mechanism in the decoder of the transformer architecture. It allows the decoder to attend to different parts of the encoded input sequence when generating the output sequence.  Unlike self-attention, which focuses on relationships within a single sequence (either input or output), cross-attention connects the decoder and encoder.

The process begins with three sets of matrices: Queries ($Q_2$), Keys ($K_2$), and Values ($V_2$).  Critically, the queries are derived from the decoder's *current* layer's output (often after a self-attention operation and denoted as $S$), while the keys and values originate from the *encoder's* final layer output (denoted as $E$).  This is where the "cross" in cross-attention comes from.

These matrices are derived using linear transformations:

$Q_2 = W_{Q_2} S$
$K_2 = W_{K_2} E$
$V_2 = W_{V_2} E$

Where $W_{Q_2}$, $W_{K_2}$, and $W_{V_2}$ are learned weight matrices specific to the cross-attention operation.

Next, the attention weights are calculated.  This begins by performing a dot-product attention operation between the queries and keys:

$Attention(Q_2, K_2, V_2) = \text{softmax}(\frac{Q_2 K_2^T}{\sqrt{d_k}}) V_2$

Here, $d_k$ is the dimensionality of the keys (and queries), and the scaling factor $\frac{1}{\sqrt{d_k}}$ is used for stability during training. The softmax function normalizes the attention weights to represent a probability distribution over the input sequence.

The "multi-head" aspect involves performing this attention mechanism multiple times with different learned linear transformations for $Q_2$, $K_2$, and $V_2$.  This allows the model to capture different aspects of the relationship between the input and output sequences. The output of each head is then concatenated and projected through a final linear layer to produce the overall multi-head cross-attention output.


Therefore the full process can be represesnted as:

$$
\text{MultiHead}(Q_2, K_2, V_2) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O
$$

where for each head,

$$
\text{head}_i = \text{Attention}(Q_2 W^Q_i, K_2 W^K_i, V_2 W^V_i)
$$

The resulting context vector from the multi-head cross attention is then typically combined with the decoder's self-attention output and passed through a feed-forward network.

## First Decoder Layer

The first decoder layer receives input from two primary sources: the output of the previous decoder layer (or the input embeddings for the first layer) and the output of the encoder stack.  Within the decoder layer, the following operations occur sequentially:

1. **Masked Multi-Head Self-Attention:** This operation attends to the input sequence, considering only the preceding tokens.  The masking prevents the model from "looking ahead" at future tokens during training, mimicking the way a human would generate a sequence word by word.  This layer receives a sequence of token embeddings. Let's represent this input as $H = [h_1, h_2, ..., h_T]$, where $h_i$ represents the embedding for the $i$-th token and $T$ is the sequence length.  This layer outputs a new sequence of vectors, say $S = [s_1, s_2, ..., s_T]$.

2. **Add & Layer Norm:** The output $S$ of the self-attention layer is then added to the original input $H$ (residual connection). This helps with gradient flow during training. The result is then normalized using layer normalization, stabilizing the training dynamics. This can be represented as:

$$
H' = \text{LayerNorm}(H + S)
$$

3. **Multi-Head Cross-Attention:** This operation allows the decoder to attend to the encoder's output, effectively incorporating information from the input sequence. This layer takes two inputs: the output $H'$ from the previous step and the encoder's output, typically denoted as $E = [e_1, e_2, ..., e_{T'}]$, where $e_i$ represents the encoder's representation of the $i$-th input token and $T'$ is the input sequence length. The output of this layer is another sequence of vectors, say $C = [c_1, c_2, ..., c_T]$.

4. **Add & Layer Norm:** Similar to step 2, the output $C$ of the cross-attention is added to $H'$ and then layer normalized:

$$
H'' = \text{LayerNorm}(H' + C)
$$

5. **Feed-Forward Network:** This layer applies a fully connected feed-forward network to each vector in the sequence $H''$ independently. This network typically consists of two linear transformations with a ReLU activation function in between.  The output of this layer is the final output of the decoder layer, which is then passed to the next decoder layer or used for prediction in the final layer.

## Number of Parameters

* **Masked Multi-Head Attention:** ~1 million parameters
* **Multi-Head Cross Attention:** ~1 million parameters
* **Feed Forward Network (FFN):** ~2 million parameters. This is calculated as follows:
    The FFN has two linear transformations. The first expands the dimensionality from 512 to 2048, and the second reduces it back to 512.  Additionally, there's a bias term for each output neuron in both layers.
    $$ \text{FFN Parameters} = (512 \times 2048 + 2048) + (2048 \times 512 + 512)$$
    $$ = 2 \times (512 \times 2048) + 2048 + 512 $$
    $$ \approx 2 \times 10^6 $$
* **Total per decoder layer:** ~4 million parameters (sum of the above).

## Decoder Output

The output from the topmost decoder layer, let's denote it as $O$, undergoes a linear transformation using a weight matrix $W_D$.  The dimensions of $O$ are $T \times d_{model}$, where $T$ is the sequence length and $d_{model}$ is the model dimension (typically 512).  $W_D$ has dimensions $d_{model} \times |V|$, where $|V|$ is the vocabulary size. The resulting matrix, let's call it $L$, will therefore have dimensions $T \times |V|$.  Each row in $L$ corresponds to a position in the output sequence, and each column represents a logit score for each word in the vocabulary. This can be represented as:

$$ L = O W_D $$

The matrix $L$ then has a softmax function applied to each row independently. This converts the logits into probabilities, producing a probability distribution over the vocabulary for each position in the output sequence.  This gives us the matrix $P$, also of size $T \times |V|$.  This operation can be expressed as:

$$ P_{t,v} = \frac{e^{L_{t,v}}}{\sum_{v'=1}^{|V|} e^{L_{t,v'}}} $$

where $P_{t,v}$ represents the probability of the $v$-th word in the vocabulary being at the $t$-th position in the output sequence.

This final probability distribution $P$ is used to predict the next word in the generated sequence during inference, typically by selecting the word with the highest probability at each time step. The matrix  $W_D$, due to its size, contributes a substantial number of parameters to the overall model (approximately $512 \times |V|$, which can be in the tens of millions depending on the vocabulary size). This transformation from the decoder's output to word probabilities is crucial for generating text.

## Positional Encoding

Positional encoding is crucial for transformers because self-attention mechanisms are permutation-invariant, meaning they don't inherently understand word order.  Therefore, positional information must be explicitly added to the input embeddings.  Several approaches could be considered:

* **Constant Vector:**  Assigning a constant vector $p_j$ to each position $j$ is too simplistic and wouldn't allow the model to differentiate effectively between different positions.

* **One-Hot Encoding:** Representing each position $j$ with a one-hot vector is another option.  However, this doesn't capture the relative distances between words.  The Euclidean distance between any two one-hot vectors would be $\sqrt{2}$, regardless of their positions in the sentence.

* **Learned Embeddings:** Learning an embedding for each possible position is possible, but becomes impractical for long sequences and doesn't generalize well to sentences longer than those seen during training.  This approach wouldn't be suitable for dynamic sentence lengths.

The solution adopted by transformers is **sinusoidal positional encoding**.  This method embeds a unique pattern of features for each position $j$, allowing the model to attend by relative position.  The encoding function is defined as:

$$
PE_{(j, i)} = \begin{cases}
\sin \left( \frac{j}{10000^{\frac{2i}{d_{model}}}} \right) & \text{if } i \text{ is even} \\
\cos \left( \frac{j}{10000^{\frac{2i-1}{d_{model}}}} \right) & \text{if } i \text{ is odd}
\end{cases}
$$

where:

* $j$ is the position of the word in the sequence.
* $i$ is the dimension of the positional encoding vector (ranges from 0 to $d_{model} - 1$).
* $d_{model}$ is the dimension of the word embeddings (typically 512).

This function generates a unique vector $p_j$ for each position $j$.  The alternating sine and cosine functions create a pattern that allows the model to learn relative positional information.  This approach has the advantage of generalizing to unseen sequence lengths.

Visualizing the positional encoding matrix as a heatmap, where rows represent positions ($j$) and columns represent dimensions ($i$), reveals distinct patterns for each position.  The first word in any sentence will always have the same positional encoding $p_0$, characterized by alternating 0s and 1s when visualized as a heatmap.  This alternating pattern is specifically produced by the sinusoidal function when $j=0$.  For subsequent positions ($j > 0$), the sinusoidal function generates increasingly complex patterns that encode relative positional information. This allows the model to distinguish between words at different positions, even if the absolute positions are beyond what it encountered during training.

## Distance Matrix and Positional Encoding Properties

The distance matrix for word positions in a sentence reveals a specific pattern: the distance increases as we move left or right from the main diagonal (representing the distance of a word from itself), and this pattern is symmetric around the center of the sentence.  This characteristic is important for capturing relationships between words based on their relative positions.

Consider the example sentence "I enjoyed the film transformer". Its distance matrix is:

|         | I      | Enjoyed   | the  | film  | transformer |
|---------|--------|-----------|------|-------|-------------|
| **I**   | 0      | 1         | 2    | 3     | 4           |
| **Enjoyed** | 1      | 0         | 1    | 2     | 3           |
| **the**  | 2      | 1         | 0    | 1     | 2           |
| **film** | 3      | 2         | 1    | 0     | 1           |
| **transformer** | 4      | 3         | 2    | 1     | 0           |

A key question is whether different positional encoding methods preserve this distance relationship.

One-hot encoding, while a simple method for representing categorical variables, fails to capture this property.  The Euclidean distance between any two distinct one-hot vectors is always $\sqrt{2}$, regardless of the words' positions in the sentence.  This constant distance means the positional information encoded is not meaningfully related to the actual distances between words.

$$
\text{Distance}(v_i, v_j) = \sqrt{\sum_{k=1}^{n} (v_{ik} - v_{jk})^2} = \sqrt{2} \quad \text{for } i \neq j
$$
where $v_i$ and $v_j$ are one-hot vectors representing different positions.

The sinusoidal positional encoding, however, is designed to incorporate this relative distance information. The use of sine and cosine functions with varying frequencies allows for the encoding of different positional relationships. While the exact relationship between the positional encoding vectors and the distance matrix isn't a direct linear mapping, the sinusoidal encoding allows the model to learn and represent relative positions effectively.  This is visually confirmed by plotting the positional encoding vectors, which reveals distinct patterns corresponding to different positions and relative distances.

## Transformer Architecture (Layers and Gradient Flow)

The transformer architecture employs a deep network structure, with the encoder and decoder each comprised of multiple identical layers.  The encoder layer contains one attention layer and two hidden layers (feed-forward networks), while the decoder layer has two attention layers (one masked self-attention and one cross-attention) and two hidden layers.  This deep architecture (42 layers in the original paper) necessitates mechanisms to ensure proper gradient flow during training and to accelerate the learning process.

To address the vanishing gradient problem often encountered in deep networks, residual connections are employed around each attention and feed-forward sublayer.  These connections allow gradients to bypass the transformations within these sublayers, facilitating easier propagation to earlier layers. Mathematically, a residual connection can be represented as:

$$
\text{Output} = \text{Sublayer}(\text{Input}) + \text{Input}
$$

This addition of the original input to the sublayer output ensures that a portion of the gradient is directly passed back during backpropagation.

Furthermore, layer normalization is used to stabilize and speed up training. Unlike batch normalization, which normalizes activations across a batch of samples, layer normalization normalizes across the features within a single layer.  This makes layer normalization less sensitive to batch size, a crucial advantage for training with variable-length sequences or small batch sizes.  The layer normalization operation can be described as follows:

1. **Calculate mean:**  $\mu = \frac{1}{H} \sum_{i=1}^{H} x_i$ , where $H$ is the number of hidden units in the layer and $x_i$ is the activation of the $i$-th unit.
2. **Calculate standard deviation:** $\sigma = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2}$
3. **Normalize:** $\hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$, where $\epsilon$ is a small constant for numerical stability.
4. **Scale and shift:** $y_i = \gamma \hat{x_i} + \beta$, where $\gamma$ and $\beta$ are learnable parameters that allow the network to restore the representational power potentially lost during normalization.

These layer normalization operations are applied after each residual connection, contributing to a more stable and efficient training process. The combination of residual connections and layer normalization is crucial for enabling the successful training of very deep transformer architectures.

## The Complete Layer (Encoder)

The complete encoder layer consists of two main sublayers: a multi-head attention block and a position-wise feed-forward network.  Both of these sublayers employ residual connections and layer normalization.

### Multi-Head Attention Block

This block performs scaled dot-product attention multiple times in parallel (the "heads"), then concatenates the results and projects them linearly.  Within each head:

1. **Linear Projections:**  The input $X$ is projected into query ($Q$), key ($K$), and value ($V$) matrices using learned weight matrices $W_Q$, $W_K$, and $W_V$ respectively.

    $Q = X W_Q$
    $K = X W_K$
    $V = X W_V$


2. **Scaled Dot-Product Attention:**  Attention weights are calculated by taking the dot product of the query matrix with the transpose of the key matrix, scaling it down by the square root of the key dimension ($d_k$) to prevent vanishing gradients, applying a softmax function to normalize the weights, and finally multiplying the result with the value matrix.

    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$

3. **Concatenation and Linear Projection:** The outputs of all attention heads are concatenated and then projected linearly using another learned weight matrix $W_O$.

    $$
    \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O
    $$

### Position-wise Feed-Forward Network

This network consists of two linear transformations with a ReLU activation in between. It is applied independently to each position in the sequence.

$$
\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2
$$

### Residual Connections and Layer Normalization

Both the multi-head attention block and the feed-forward network utilize residual connections and layer normalization.  The input to each sublayer is added to the output of the sublayer (residual connection), and then layer normalization is applied to the sum.  This helps with gradient flow and training stability. Specifically:

$$
\text{LayerNorm}(\text{Sublayer}(x) + x)
$$
Where "Sublayer" can be either the multi-head attention block or the feed-forward network.

## The Transformer Architecture (Overall)

The Transformer architecture eschews recurrence and instead relies entirely on an attention mechanism to draw global dependencies between input and output.  It's trained with learned embeddings, unlike some models that use pre-trained embeddings.

The model starts with input embeddings for each word in the source sequence. Learned positional encodings are added to these embeddings to provide information about word order, crucial because the attention mechanism itself is permutation-invariant.

The encoder consists of a stack of identical layers.  Each layer comprises two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.  Residual connections around each of these sub-layers are employed, followed by layer normalization.  This structure allows for easier gradient flow during training and can help mitigate vanishing gradient issues. The output of the final encoder layer provides a contextualized representation of the entire input sequence.

The decoder also consists of a stack of identical layers.  Each decoder layer includes the two sub-layers from the encoder (multi-head self-attention and feed-forward network) but adds a third sub-layer: a multi-head cross-attention mechanism.  This cross-attention layer allows the decoder to focus on relevant parts of the encoded input sequence when generating the output sequence.  Similar to the encoder, residual connections and layer normalization are applied after each sub-layer in the decoder.

Crucially, the decoder operates autoregressively. During training, teacher forcing can be employed, where the ground-truth output is provided as input to the decoder.  However, during inference, the decoder generates the output sequence one token at a time, with each previously generated token becoming input for generating the next one.  The decoder's self-attention mechanism is masked to prevent it from attending to future tokens in the output sequence during both training and inference.  This masking ensures that the prediction for a given position depends only on the preceding tokens.

The output of the final decoder layer is then projected to a logits vector, the dimension of which is equivalent to the output vocabulary size.  A softmax function is applied to these logits to produce a probability distribution over the vocabulary.  The token with the highest probability is then selected as the output for that position in the generated sequence.


## Review Questions

1. **Explain the purpose of the mask in the decoder's self-attention mechanism. How is it implemented, and what is its impact on the attention weights?**  Your answer should cover the structure of the mask matrix and its effect on preventing information leakage from future tokens.

2. **Describe the difference between self-attention and cross-attention in the transformer architecture. What are the inputs to each, and how do they contribute to the overall functioning of the encoder and decoder?**  Focus on where the queries, keys, and values come from in each case.

3. **Teacher forcing is a crucial technique during transformer training.  Explain how it works and why it's beneficial. What happens during inference when teacher forcing is disabled?**  Your explanation should include the difference in conditional probabilities with and without teacher forcing.

4. **Why are positional encodings necessary in the transformer architecture? Discuss the limitations of alternative approaches like one-hot encoding and learned embeddings, and explain how sinusoidal positional encoding addresses these limitations.** Be sure to explain the properties of the sinusoidal function and how it represents positional information.

5. **Describe the complete flow of information through a single encoder layer and a single decoder layer in a transformer. What are the sub-layers involved, and how are residual connections and layer normalization incorporated?**  Your response should include the order of operations and the mathematical representations of the residual connections and layer normalization.

6. **The output of the transformer decoder is a probability distribution over the vocabulary.  Explain the steps involved in transforming the output of the final decoder layer into this probability distribution.  What role does the weight matrix $W_D$ play, and what are its dimensions?**  Your answer should explain the linear transformation and softmax operation, including the dimensions of the matrices involved.

7.  **Why are residual connections and layer normalization used in the transformer architecture? What problem do they address, and how do they contribute to the training process?** Make sure to differentiate between layer normalization and batch normalization.

8. **Explain the "multi-head" aspect of the attention mechanism.  How does it work, and what are its benefits?** Your answer should describe how the multiple heads operate and how their outputs are combined.


9. **Considering the distance matrix of word positions in a sentence, explain why one-hot encoding is insufficient for representing positional information.  How does sinusoidal positional encoding address this shortcoming?**  Discuss the Euclidean distance between one-hot vectors and how the sinusoidal encoding captures relative distances.


10. **Given the equation for sinusoidal positional encoding ($PE_{(j,i)}$), what is the characteristic pattern of the positional encoding $p_0$ for the first word ($j=0$) in any sentence?  How does this pattern relate to the sinusoidal function?**  Describe the alternating 0s and 1s visualized in the heatmap.
