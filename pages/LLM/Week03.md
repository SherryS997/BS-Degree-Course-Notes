---
title: Language Models and GPT 
---

## Language Modelling

Language modeling is a fundamental task in natural language processing (NLP) that focuses on predicting the likelihood of a sequence of words in a given language. It aims to capture the statistical regularities and underlying structure of a language, allowing us to understand how words are related and how they combine to form meaningful sentences. 

### Core Concepts

1. **Vocabulary (V):**  A set of all unique words in the language under consideration. This serves as the building block for constructing sentences. 
2. **Sentence Representation:** A sentence is represented as a sequence of words, where each word belongs to the vocabulary: $X_1, X_2, ..., X_n$, where $X_i ∈ V$.
3. **Probability Distribution:** The core goal of language modeling is to define a probability distribution over all possible sequences of words in the vocabulary. This distribution captures the likelihood of observing a particular sentence or sequence of words.
4. **Language Model Function:**  A language model can be formalized as a function that takes a sequence of words as input and outputs a probability score between 0 and 1, indicating the likelihood of that sequence. Mathematically: $f: (X_1, X_2, ..., X_n) → [0, 1]$.

### Examples

Let's illustrate these concepts with a simple example:

- **Vocabulary (V):** `{'an', 'apple', 'ate', 'I'}`
- **Possible Sentences:** 
    - `An apple ate I`
    - `I ate an apple`
    - `I ate apple`
    - `an apple` 
- **Probability:** Intuitively, some of these sentences are more probable than others. For instance, "I ate an apple" is likely to be more probable than "An apple ate I" based on the grammatical structure and common usage of English. 
- **Language Model Function:** A language model would assign a higher probability score to the sentence "I ate an apple" than to "An apple ate I." 

### The Importance of Probability

The probability assigned to a sequence reflects how likely it is to occur in a given language. This probability can be derived from a large collection of text data (corpus) by observing how frequently various word sequences appear. 

### Probability Calculation: The Chain Rule

A fundamental concept in language modeling is the chain rule of probability. It allows us to decompose the probability of an entire sequence into the probabilities of individual words, conditioned on the preceding words in the sequence.  This captures the dependencies between words in a sentence.

The chain rule states:

$$
P(x_1, x_2, ..., x_T) = \prod_{i=1}^{T} P(x_i | x_1, ..., x_{i-1})
$$

- **Interpretation:** The probability of observing the sequence $x_1, x_2, ..., x_T$ is equal to the product of the conditional probabilities of each word $x_i$, given the preceding words $x_1, ..., x_{i-1}$.


### Naive Approach: Independence Assumption

A simplified approach to language modeling is to assume that the words in a sequence are independent of each other.  This assumption ignores the contextual relationships between words. While simplistic, it offers a starting point for understanding language modeling.

Under this independence assumption, the probability of a sequence becomes:

$$
P(x_1, x_2, ..., x_T) = \prod_{i=1}^{T} P(x_i)
$$

- **Interpretation:** The probability of the sequence is simply the product of the probabilities of each individual word occurring independently.


### The Need for Context: Dependence on Previous Words

However, the independence assumption is often unrealistic. Words are rarely independent. The meaning and likelihood of a word depend heavily on the surrounding words.

**Example:**

- **Sentence 1:** `I enjoyed reading a book`
- **Sentence 2:** `I enjoyed reading a thermometer`

In these examples, the presence of "enjoyed" makes the word "book" significantly more probable than "thermometer" in the context of the sentence. This illustrates that words are strongly influenced by their context. 


### Estimation of Conditional Probabilities

The core challenge in language modeling is to accurately estimate the conditional probabilities $P(x_i | x_1, ..., x_{i-1})$.  How can we determine the probability of a word given its preceding words?  

This is where various language modeling techniques come into play. These techniques involve utilizing large amounts of text data and employing statistical or machine learning methods to learn the relationships between words and their contexts. 


###  Autoregressive Models

One powerful approach to language modeling is the use of autoregressive models. These models learn to predict the next word in a sequence based on the preceding words. They are particularly well-suited for capturing the dependencies between words in a sequence.

**Key Idea:** Autoregressive models represent the conditional probabilities $P(x_i | x_1, ..., x_{i-1})$ as parameterized functions, typically neural networks.  These functions are trained on a large corpus of text data to learn the underlying patterns of language.


By understanding these core concepts and the challenges involved in estimating conditional probabilities, we pave the way to explore more advanced language modeling techniques such as the Causal Language Models (CLMs) and the GPT architecture which are discussed in the following sections. 

## Causal Language Modelling (CLM)

Causal Language Modelling (CLM) is a fundamental approach in language modeling that leverages the chain rule of probability to model the sequential nature of language. The core idea is to predict the probability of the next word in a sequence given the preceding words. This approach is crucial for tasks like text generation, where we want the model to generate text sequentially, one word at a time.

**Core Principles:**

1. **Sequential Prediction:** CLM focuses on predicting the probability of the current word $x_i$ given all the previous words in the sequence ($x_1, x_2, ..., x_{i-1}$).
2. **Autoregressive Nature:** The model is autoregressive, meaning its predictions depend on its own previous outputs. This allows it to generate text incrementally.
3. **Chain Rule Application:** CLM utilizes the chain rule of probability to decompose the joint probability of a sequence into a product of conditional probabilities.


**Mathematical Formulation:**

The probability of a sequence of words $x_1, x_2, ..., x_T$ in CLM is calculated as follows:

$$
P(x_1, x_2, ..., x_T) = \prod_{i=1}^{T} P(x_i | x_1, ..., x_{i-1}) 
$$


**Objective:**

The objective of CLM is to find a parameterized function $f_θ$ that can accurately model the conditional probabilities $P(x_i | x_1, ..., x_{i-1})$. This function, often implemented as a neural network (like a transformer), learns to capture the relationships and dependencies between words in a sequence.

$$
P(x_i | x_1, ..., x_{i-1}) = f_θ(x_i | x_1, ..., x_{i-1})
$$


**Why is CLM important?**

- **Text Generation:** CLM is crucial for generating text, as it enables the model to produce text sequentially, one word at a time. The model predicts the most likely next word given the previously generated words, effectively creating a coherent and contextually relevant text sequence.
- **Language Understanding:** By learning to predict the next word, CLM models implicitly learn to understand the relationships and dependencies between words, forming a basis for understanding the structure and semantics of language.
- **Downstream Tasks:** CLM provides a strong foundation for many downstream NLP tasks, such as machine translation, text summarization, and question answering. The learned representations can be further fine-tuned for specific tasks.

### Transformer Application in CLM: A Detailed Look

1. **Input Embedding:** The input sequence of words (x_1, x_2, ..., x_{i-1}) is first converted into a sequence of embedding vectors. Each word is mapped to a dense vector representation that captures its semantic meaning and relationship to other words in the vocabulary.
2. **Positional Encoding:**  Since the transformer architecture doesn't inherently understand the order of words, positional encoding is added to the embedding vectors. This provides information about the position of each word in the sequence.
3. **Decoder Layers (Transformer Blocks):** The sequence of embedded and positionally encoded words is then fed into a stack of decoder layers, also known as transformer blocks. Each decoder layer consists of two sub-layers:
    - **Masked Multi-Head Self-Attention:** This crucial component allows the model to weigh the importance of different words in the input sequence when predicting the next word. The "masked" part is critical for CLM because it ensures that the model only attends to previous words in the sequence, preventing it from "peeking" at future words. This is implemented using a mask matrix, similar to the example shown earlier.
        - **Query (Q), Key (K), Value (V) Matrices:** The input sequence is projected into three matrices: Q, K, and V. 
        - **Scaled Dot-Product Attention:** The attention weights are calculated using the dot product of the query and key matrices, scaled down by the square root of the key dimension. 
        - **Softmax:**  The scaled dot products are then passed through a softmax function, which normalizes the weights to form a probability distribution over the input sequence.
        - **Value Matrix Multiplication:** The softmax output is then multiplied with the value matrix to obtain a weighted representation of the input sequence.
    - **Feed-Forward Neural Network (FFN):**  After self-attention, a feed-forward neural network is applied to each position in the sequence. This allows the model to learn non-linear relationships between words and refine the representation further. 
4. **Output Layer:** The final decoder layer outputs a vector for each position in the sequence. This vector represents the model's understanding of the context up to that point.
5. **Prediction:** A linear layer (often called a language modeling head) is applied to the output vector to generate a probability distribution over the vocabulary. This distribution represents the model's prediction for the next word in the sequence given the preceding context.
6. **Loss Calculation:**  During training, the model's predictions are compared to the actual next word in the sequence (the ground truth). A loss function (e.g., cross-entropy loss) is used to quantify the difference between the predicted and actual probabilities. The model's parameters are then updated using an optimization algorithm (e.g., Adam) to minimize this loss.


**In essence, the transformer in CLM learns to predict the next word in a sequence by attending to the relevant words in the past context, using its multi-head self-attention mechanism. The decoder layers progressively refine the representation of the input sequence, allowing the model to capture long-range dependencies and generate highly probable language.** 

### Masked Multi-Head Attention

Masked Multi-Head Attention is a crucial component of the GPT architecture, responsible for enabling the model to attend to different parts of the input sequence while preventing it from "peeking" into future tokens. This is essential for maintaining the autoregressive nature of the model during training. 

Here's a breakdown of the process and its components:

1. Input Sequence

    The input to Masked Multi-Head Attention is a sequence of tokens represented as word embeddings. For example:

    ```plaintext
    <go> at the bell labs hammering ...... bound ..... devising a new <stop> 
    ```

    Each token is transformed into a vector of dimension $d_{model}$ (768 in GPT-1).

2. Creating Query (Q), Key (K), and Value (V) Matrices

    - The input embeddings are linearly projected into three different matrices: Query (Q), Key (K), and Value (V). 
    - Each of these matrices has a dimension of $(T, d_k)$, where $T$ is the sequence length and $d_k$ is the dimension of the key/query/value vectors (typically $d_{model}$ / number of attention heads).
    - The linear projections are performed using learned weight matrices $W_Q$, $W_K$, and $W_V$:

    $$
    Q = XW_Q \\
    K = XW_K \\
    V = XW_V
    $$
    Where $X$ represents the input embeddings.

3. Calculating Scaled Dot-Product Attention

    - The scaled dot-product attention mechanism calculates the attention weights between different tokens in the sequence.
    - It measures the relevance of each token in the sequence to the current token being processed. 
    - The formula for scaled dot-product attention is:

    $$
    \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
    $$

    - **QK<sup>T</sup>:** This calculates the dot product between the query matrix and the transpose of the key matrix. It generates a matrix of scores representing the similarity between each query and each key.
    - **Scaling by $\sqrt{d_k}$:** This helps to stabilize the gradients during training, especially when $d_k$ is large.
    - **Softmax:** This normalizes the scores into a probability distribution, where each element represents the probability of attending to a specific token.
    - **Multiplication with V:** The attention weights are multiplied with the value matrix to generate a weighted sum of the value vectors. This weighted sum represents the context-aware representation of the current token.

4. Applying the Mask

    - The mask is crucial for preventing the model from attending to future tokens during training. 
    - It is a matrix of the same dimensions as the `QK<sup>T</sup>` matrix.
    - The mask contains values of 0 for allowed connections and $-\infty$ for connections that should be masked out (i.e., connections to future tokens).
    - **Example Mask Matrix:**
    $$ M = \begin{bmatrix}
    0 & -\infty & -\infty & -\infty & -\infty \\
    0 & 0 & -\infty & -\infty & -\infty \\
    0 & 0 & 0 & -\infty & -\infty \\
    0 & 0 & 0 & 0 & -\infty \\
    0 & 0 & 0 & 0 & 0 
    \end{bmatrix} $$
    - This mask ensures that when calculating `QK<sup>T</sup>`, the connections to future tokens are effectively ignored by the softmax function (because $-\infty$ after softmax becomes 0).
    - **Applying the Mask:** The mask is added to the `QK<sup>T</sup>` matrix before applying the softmax function:
    $$
    \text{Masked Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} + M \right) V
    $$

5. Multi-Head Attention

    - GPT utilizes multiple attention heads, each focusing on different aspects of the input sequence. 
    - Each head performs the scaled dot-product attention independently.
    - The outputs of all heads are concatenated and linearly transformed to produce the final output of the multi-head attention layer.

6. Dropout

    - Dropout is applied after the softmax activation and before the matrix multiplication with V. 
    - This helps to prevent overfitting by randomly dropping out some of the connections in the attention mechanism.

7. Residual Connection and Layer Normalization

    - The output of the multi-head attention is added to the input of the layer (residual connection) and then normalized (layer normalization).
    - This helps to improve the flow of gradients during training and stabilizes the learning process.


#### Overall Process Summary

1. **Input Embedding:** Transform input tokens into embedding vectors.
2. **Linear Projections:** Project embeddings into Q, K, and V matrices.
3. **Scaled Dot-Product Attention:** Calculate attention weights based on Q and K.
4. **Mask Application:** Add the mask to `QK<sup>T</sup>` to prevent attending to future tokens.
5. **Softmax and Value Multiplication:** Normalize attention weights and generate a weighted sum of V.
6. **Multi-Head Attention:** Concatenate and transform the outputs of multiple attention heads.
7. **Residual Connection and Layer Normalization:** Stabilize training and improve gradient flow.

## Generative Pretrained Transformer (GPT)

GPT leverages the decoder-only transformer architecture for language modeling. It aims to learn the probability distribution of a sequence of tokens, predicting the likelihood of the next token given the preceding tokens. 

**Core Idea:** GPT learns to generate human-like text by predicting the next token in a sequence during pre-training, which allows it to capture intricate language patterns and relationships between words. This pre-trained model can then be fine-tuned for various downstream NLP tasks.


### GPT Pre-training

The pre-training phase is crucial for establishing a strong language understanding foundation in GPT. Here's a breakdown of the key aspects:


**Objective:** Maximize the likelihood of the sequence of tokens in a corpus.

**Loss Function:**

$$
\mathcal{L} = - \sum_{i=1}^T \log P(x_i | x_1, ..., x_{i-1}) 
$$

Where:

-  $x_i$ represents the $i$-th token in the sequence.
-  $P(x_i | x_1, ..., x_{i-1})$ is the probability of token $x_i$ given the preceding tokens in the sequence. 
-  The summation iterates through the entire sequence length $T$. 

**Dataset:** GPT-1 utilized the BookCorpus dataset, which is a collection of 7,000 unique books, encompassing approximately 1 billion words and 74 million sentences across 16 genres. This large-scale dataset is crucial for the model to learn a broad range of language patterns and styles.

**Tokenizer:** GPT-1 employed Byte Pair Encoding (BPE) as its tokenizer. BPE is a subword-level tokenizer that breaks down words into smaller units (subwords or byte pairs) based on their frequency in the training data. This approach helps handle out-of-vocabulary (OOV) words and improves the model's ability to generalize to unseen data.

**Input Representation:** Each token in the input sequence is represented as a vector with a dimensionality equal to the embedding dimension ($d_{model}$).  The model's input during training is a sequence of tokens, represented as a 3-dimensional tensor: `(batch_size, sequence_length, embedding_dimension)`.


**Training Procedure:**

1. **Tokenization:** The input text is tokenized into a sequence of tokens using BPE.
2. **Embedding:** Each token is mapped to its corresponding embedding vector.
3. **Positional Encoding:** Positional embeddings are added to the token embeddings to provide information about the position of each token in the sequence. 
4. **Transformer Decoder Blocks:** The input sequence is fed through a stack of transformer decoder blocks. Each block consists of a multi-head masked self-attention mechanism, a feed-forward neural network, and layer normalization.
5. **Output Layer:** The final decoder block's output is fed into an output layer, which predicts the probability distribution over the vocabulary for each position in the sequence.
6. **Loss Calculation:** The loss function is calculated based on the predicted probabilities and the actual target tokens.
7. **Backpropagation and Optimization:** The model's parameters are updated using backpropagation and an optimization algorithm (Adam in GPT-1) to minimize the loss function.


### GPT Architecture

The GPT architecture is based on the transformer decoder model with modifications for language modeling. Let's delve into the core components:

**Transformer Decoder Blocks:**

GPT employs a stack of 12 transformer decoder blocks. Each block comprises the following sub-layers:

1. **Masked Multi-head Self-Attention:** This sub-layer allows the model to attend to different parts of the input sequence and weigh their importance in determining the probability of the next token. The "masked" part ensures that the model only attends to previous tokens and prevents it from "peeking" into future tokens during training.  
2. **Position-wise Feed-Forward Networks (FFN):** After the self-attention, a feed-forward network is applied to each position in the sequence. This network consists of two linear transformations with a non-linear activation function (GELU in GPT-1) in between. It enhances the model's ability to capture complex relationships between tokens.
3. **Layer Normalization:** Layer normalization is applied after each sub-layer to stabilize the training process and improve the model's performance.
4. **Residual Connections:**  Residual connections are used to connect the output of each sub-layer to its input, allowing the model to learn identity mappings and aiding in training deeper networks.


**Other Key Aspects:**

- **Context Size:** The maximum sequence length (context) that the model can process is 512 tokens.
- **Number of Attention Heads:** 12 attention heads are used in each multi-head attention sub-layer. 
- **Hidden Size:** The hidden size, also referred to as the model dimension, is 768. This refers to the dimensionality of the embeddings and the hidden states within the transformer blocks.
- **Feed-Forward Network Hidden Size:** Each FFN has an intermediate hidden size of 3072 (4 times the model dimension).
- **Activation Function:** The GELU activation function is used in the FFN layers.


### Number of Parameters in GPT-1


Let's break down the parameter counts for the different components of GPT-1:


**1. Token Embeddings:**

- The embedding layer maps each token in the vocabulary to a 768-dimensional vector.
- Number of parameters: `|Vocabulary| * embedding_dimension` = `40478 * 768` ≈ **31 million**


**2. Positional Embeddings:**

- Positional embeddings are learned parameters that encode the position of each token in the sequence.
- Number of parameters: `sequence_length * embedding_dimension` = `512 * 768` ≈ **0.3 million**


**3. Attention Parameters per Block:**

- **Query, Key, and Value Matrices:** For each attention head, there are three weight matrices: `W_Q`, `W_K`, and `W_V`. Each matrix has dimensions `embedding_dimension * head_dimension`.
- **Output Projection:**  An output projection matrix `W_O` projects the concatenated attention outputs to the embedding dimension.
- Number of parameters per attention head: `3 * (embedding_dimension * head_dimension) + (embedding_dimension * embedding_dimension)` ≈ `3 * (768 * 64) + (768 * 768)` ≈ **1.7 million**. 
- For 12 attention heads: `12 * 1.7 million` ≈ **20.4 million**.
- For all 12 blocks: `12 * 20.4 million` ≈ **244.8 million**.

**4. FFN Parameters per Block:**

- Each FFN has two linear transformations with a hidden layer size of 3072.
- Number of parameters: `2 * (embedding_dimension * FFN_hidden_size) + FFN_hidden_size + embedding_dimension` ≈ `2 * (768 * 3072) + 3072 + 768` ≈ **4.7 million**.
- For all 12 blocks: `12 * 4.7 million` ≈ **56.4 million**.


**Total Number of Parameters:**

Summing up the parameter counts for the different components: 

**~117 million**

## Fine-tuning GPT

Fine-tuning involves adapting a pre-trained GPT model to a specific downstream task by making minimal changes to its architecture. The primary goal is to leverage the general language understanding learned during pre-training and specialize it for a particular application. This process typically involves adjusting the model's input and output layers while retaining the core transformer architecture.

### Input Modifications for Fine-tuning

During fine-tuning, the input sequence is often modified to include task-specific tokens. These tokens provide contextual information to the model about the task at hand. For instance:

- **Classification tasks:**  We might add special start (`<s>`) and end (`</s>`) tokens to demarcate the input sequence for classification.
- **Sequence labeling:** We might incorporate tokens that represent the beginning and end of entities or segments within the input sequence. 
- **Question answering:** We could use tokens to distinguish between questions and context paragraphs.


### Output Layer Modification: Replacing the Language Modeling Head

The pre-trained GPT model is designed for language modeling, where the output is the probability distribution over the vocabulary for the next token.  For fine-tuning to a different task, we replace this language modeling head with a task-specific output layer. This new layer is typically a linear transformation followed by a softmax function, creating a probability distribution over the desired output space.

- **Classification tasks:** The output layer would generate a probability distribution over the classes (e.g., positive/negative for sentiment analysis).
- **Regression tasks:** The output layer could directly produce a continuous value (e.g., predicting a numerical rating or score).
- **Sequence labeling:** The output layer would predict a label for each token in the input sequence.


### Fine-tuning Objective Function

The fine-tuning process aims to optimize a new objective function tailored to the specific downstream task. This objective function is often a loss function that measures the discrepancy between the model's predictions and the true labels in the training data.

- **Classification tasks:** The cross-entropy loss function is commonly used. It measures the difference between the model's predicted probability distribution over classes and the true class label.
  $$
  \mathcal{L}_{CE} = - \sum_{i} y_i \log(\hat{y}_i)
  $$
  where $y_i$ is the true label (one-hot encoded) and $\hat{y}_i$ is the predicted probability for class $i$.

- **Regression tasks:** Mean squared error (MSE) is a common choice for regression problems. It measures the squared difference between the predicted and true values.
  $$
  \mathcal{L}_{MSE} = \frac{1}{N} \sum_{i} (y_i - \hat{y}_i)^2
  $$
  where $y_i$ is the true value and $\hat{y}_i$ is the predicted value.


### Example: Fine-tuning for Sentiment Analysis

1. **Input Modification:** We might add `<s>` and `</s>` tokens to the input sequence. 
   ```
   <s> Wow, India has now reached the moon. </s>
   ```

2. **Output Layer Modification:**  Replace the language modeling head with a linear layer that projects the final hidden state ($h_{12}$) to two output neurons representing the positive and negative classes.
   
3. **Objective Function:** The cross-entropy loss would be used to measure the difference between the model's predicted sentiment probability and the true sentiment label.

4. **Training:** The model is trained on a dataset of sentences paired with their corresponding sentiment labels. The gradients are calculated based on the cross-entropy loss, and the model parameters are updated to minimize this loss.


### Considerations during Fine-tuning

- **Learning Rate:**  A lower learning rate is often used during fine-tuning compared to pre-training to prevent drastic changes to the pre-trained weights.
- **Number of Training Steps:** Fine-tuning typically requires fewer training steps than pre-training, as the model already has a strong foundation.
- **Data Augmentation:** Augmenting the training data can help improve the model's generalization capabilities.
- **Hyperparameter Tuning:**  Experiment with different hyperparameters (e.g., learning rate, batch size, number of training epochs) to optimize performance on the target task.

## Downstream Tasks using GPT

### Sentiment Analysis

- **Goal:** Classify a piece of text as expressing a positive, negative, or neutral sentiment. 
- **Input:** A sequence of tokens representing the text.
- **Output:** A predicted sentiment label (e.g., positive, negative, neutral).
- **Example:**
    - **Input:** "The movie was absolutely fantastic!"
    - **Output:** Positive.
- **Fine-tuning Process:**
    1. Add special start (`<s>`) and end (`</s>`) tokens to the input sequence.
    2. Replace the language modeling head with a classification head ($W_y$) that has a softmax layer to output probabilities over the sentiment classes.
    3. Train the model on a dataset of text samples labeled with their corresponding sentiment.
    4. The model learns to associate specific word patterns and sentence structures with different sentiments.

### Textual Entailment/Contradiction

- **Goal:** Determine the relationship between a given text (premise) and a hypothesis. The relationship can be entailment (hypothesis is true given the premise), contradiction (hypothesis is false given the premise), or neutral (no relationship).
- **Input:** Two sequences of tokens, one for the premise and one for the hypothesis, separated by a delimiter token ($).
- **Output:** A label indicating the relationship between the premise and hypothesis (e.g., entailment, contradiction, neutral).
- **Example:**
    - **Premise:** "The cat sat on the mat."
    - **Hypothesis:** "The cat is on a surface."
    - **Output:** Entailment.
- **Fine-tuning Process:**
    1. Concatenate the premise and hypothesis sequences with a delimiter token ($).
    2. Replace the language modeling head with a classification head ($W_y$) that outputs probabilities over the entailment relationship classes. 
    3. Train the model on a dataset of premise-hypothesis pairs labeled with their relationship.
    4. The model learns to identify the semantic relationship between the premise and hypothesis.


### Multiple Choice Question Answering

- **Goal:** Answer a multiple-choice question by selecting the most appropriate option.
- **Input:** A question and a set of answer choices.
- **Output:** The index of the chosen answer.
- **Example:**
    - **Question:** "What is the capital of France?"
    - **Choices:** (A) London, (B) Paris, (C) Berlin, (D) Rome
    - **Output:** (B)
- **Fine-tuning Process:**
    1. Concatenate the question and each answer choice separately, creating multiple input sequences. 
    2. Replace the language modeling head with a classification head ($W_y$) that outputs probabilities over the answer choices.
    3. Train the model on a dataset of question-answer choice pairs labeled with the correct answer.
    4. The model learns to associate the question with the most relevant answer choice.


### Text Generation

- **Goal:** Generate creative and coherent text based on a given prompt or context.
- **Input:** A prompt or starting sequence of tokens.
- **Output:** A continuation of the sequence generated by the model. 
- **Example:**
    - **Input:** "Once upon a time, in a faraway land..."
    - **Output:** "...there lived a brave knight who..."
- **Fine-tuning Process:**
    1. The model is fine-tuned using the same pre-training objective (language modeling) but often with a different dataset that focuses on diverse and creative text samples. 
    2. During generation, the model receives the prompt as input and uses its learned knowledge to predict the next token in the sequence, iteratively extending the text.
    3. Sampling techniques (e.g., nucleus sampling, top-k sampling) are used to control the randomness and creativity of the generated text.

## Review Questions

**Conceptual Understanding:**


1. **What is the primary goal of language modeling? How does it relate to the concept of a vocabulary and sentence representation?** (Assesses understanding of core concepts and their interconnections).
2. **Explain the chain rule of probability in the context of language modeling. Why is it important for capturing language structure?** (Tests understanding of the chain rule and its significance).
3. **What is the independence assumption in language modeling? Why is it often unrealistic? Provide an example.** (Evaluates comprehension of the naive approach and the need for context).
4. **Describe the role of autoregressive models in language modeling. How do they address the challenge of estimating conditional probabilities?** (Checks understanding of autoregressive models and their relevance to the task).
5. **What are the core principles of Causal Language Modeling (CLM)? How does it relate to the chain rule of probability?** (Assesses understanding of CLM and its connection to the fundamental probability concept).


**Transformer and GPT:**


6. **Explain the role of Masked Multi-Head Self-Attention in the GPT architecture. Why is masking crucial for CLM?** (Focuses on a key component and its significance for the autoregressive nature).
7. **Describe the components of a Transformer Decoder Block in GPT. Explain the purpose of each component.** (Checks understanding of the core building blocks of the model).
8. **What is the objective function used during GPT pre-training? Explain the components of this function.** (Tests understanding of the model's training goal).
9. **How does Byte Pair Encoding (BPE) contribute to GPT's effectiveness?** (Evaluates comprehension of the role of tokenization).
10. **Explain the difference between positional encoding and token embedding in GPT. Why is positional encoding necessary?** (Assesses understanding of how the model represents both token identity and order).


**Fine-tuning and Downstream Tasks:**


11. **Describe the process of fine-tuning a GPT model for a specific downstream task. What aspects of the model are typically modified?** (Tests comprehension of the adaptation process).
12. **Explain how the output layer of a GPT model is modified during fine-tuning for different tasks (e.g., classification, regression).** (Evaluates understanding of how the output is adapted to different task types).
13. **What are some common considerations when fine-tuning a GPT model?** (Focuses on practical aspects of fine-tuning).
14. **Choose one of the downstream tasks discussed (e.g., sentiment analysis, textual entailment, question answering) and explain the specific steps involved in adapting GPT for that task.** (Requires application of knowledge to a specific example).
15. **Explain how GPT can be used for text generation. What are some challenges in achieving high-quality text generation?** (Checks comprehension of the text generation process and its challenges).
