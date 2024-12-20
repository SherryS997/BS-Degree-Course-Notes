---
title: Sequence-to-Sequence Models and Attention Mechanisms
---

## Overview

These notes cover the fundamentals of sequence-to-sequence (seq2seq) models, particularly focusing on Recurrent Neural Networks (RNNs) and the transformative role of attention mechanisms. We'll explore the architecture of seq2seq models, the limitations of traditional encoder-decoder RNNs, and how attention addresses these limitations. We'll delve into the details of self-attention and multi-head attention, examining their computational aspects and benefits. Finally, we'll connect these concepts to the Transformer architecture, a powerful model that leverages attention mechanisms extensively.

# Sequence-to-Sequence Models

## Definition and Scope

Sequence-to-sequence models are a class of deep learning models designed to map input sequences (e.g., sentences in one language) to output sequences (e.g., sentences in another language). They are widely used in tasks like machine translation, text summarization, speech recognition, and question answering.

## Encoder-Decoder Architecture

### Encoder RNN

- The encoder RNN processes the input sequence one element at a time.
- At each step, it updates its hidden state based on the current input and the previous hidden state.
- The final hidden state of the encoder, often called the **context vector**, encapsulates the information from the entire input sequence.

### Decoder RNN

- The decoder RNN takes the context vector as its initial hidden state.
- It generates the output sequence element by element, conditioning on the context vector and the previously generated outputs.

## Limitations of Traditional Encoder-Decoder RNNs

- The context vector acts as a bottleneck, as it needs to represent the entire input sequence in a fixed-size vector.
- This can lead to information loss, especially for long input sequences.
- The decoder may struggle to align words in the input and output sequences effectively.


# Attention Mechanisms

## Definition and Motivation

Attention mechanisms address the limitations of traditional encoder-decoder RNNs by allowing the decoder to focus on different parts of the input sequence at each step of the output generation process. This is achieved by computing attention weights that indicate the relevance of each input element to the current decoder state.

## Attention Mechanism: A Quick Tour

1. **RNN Encoder:**
   - The encoder produces a sequence of hidden state vectors (h0, h1, h2, ...). Each vector corresponds to a word or token in the input sequence.

2. **Decoder Access to Encoder States:**
   - The decoder has access to all the hidden state vectors generated by the encoder.

3. **Attention Mechanism Input:**
   - The attention mechanism takes these encoder hidden state vectors as input.

## Context Vector Computation

- The context vector `ct` for output `yt` at time step `t` is computed as a weighted sum of the encoder hidden states:

$$c_t = \sum_{i=1}^{n} \alpha_{ti} h_i$$

where:

- `n` is the number of words in the input sequence
- `αti` is the attention weight for the i-th input word at time step `t`
- `hi` is the hidden state of the encoder for the i-th input word

## Alignment of Words

Traditional seq2seq models without attention struggle to align words between the source and target sentences. Attention mechanisms resolve this by learning to focus on the relevant parts of the source sentence when generating each word in the target sentence.

## Computing Alignment Scores

The alignment score `αi` between the output word `yt` at time step `t` and the input word with hidden state `hi` is computed as:

$$ \alpha_i = align(y_t, h_i) = \frac{exp(score(s_{t-1}, h_i))}{\sum_{i'=1}^{n} exp(score(s_{t-1}, h_{i'}))} $$

where:

- `st-1` is the decoder's hidden state at the previous time step
- `score` is a function that computes the relevance between the decoder state and the encoder hidden state (e.g., dot product)


# Self-Attention

## Definition

Self-attention is a type of attention mechanism where the model attends to different positions within the same input sequence. It allows the model to learn relationships between different words in a sentence and capture contextual information.

## Example

Consider the sentence: "The animal didn't cross the street because it was too tired."

- Self-attention helps the model understand that "it" refers to "animal."
- If we modify the sentence to: "The animal didn't cross the street because it was congested," self-attention helps the model understand that "it" now refers to "street."

## Goal

Given a word in a sentence, self-attention aims to compute the relational score between that word and all other words in the sentence.

## Relational Score Table

| Word | The | animal | didn't | cross | the | street | because | it |
|---|---|---|---|---|---|---|---|---|
| The | 0.6 | 0.1 | 0.05 | 0.05 | 0.02 | 0.02 | 0.02 | 0.1 |
| animal | 0.02 | 0.5 | 0.06 | 0.15 | 0.02 | 0.05 | 0.01 | 0.12 |
| didn't | 0.01 | 0.35 | 0.45 | 0.1 | 0.01 | 0.02 | 0.01 | 0.03 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |


# Multi-Head Attention

## Motivation

- Similar to using multiple filters in Convolutional Neural Networks (CNNs) to learn different features, multi-head attention allows the model to capture different aspects of the relationships between words in a sentence.
- Each attention head learns to focus on a different type of relationship (e.g., syntactic, semantic).

## Mechanism

- Multi-head attention consists of multiple self-attention heads operating in parallel.
- Each head has its own set of learnable parameters (WQ, WK, WV).
- The outputs of the different heads are concatenated and then transformed through a linear layer.

## Benefits

- Captures richer contextual information.
- Allows for parallel computation.


# Transition to Transformers

## Encoder and Decoder

- The Transformer architecture is based on the encoder-decoder framework but replaces RNNs with attention mechanisms.

## Encoder Components

- **Word Embedding:** Converts words into vector representations.
- **Self-Attention:** Captures relationships between words in the input sequence.
- **Feed Forward Networks:** Processes the output of the self-attention layer.

## Decoder Components

- **Word Embedding:** Converts words into vector representations.
- **Self-Attention:** Captures relationships between words in the output sequence.
- **Encoder-Decoder Attention:** Allows the decoder to attend to different parts of the input sequence.
- **Feed Forward Networks:** Processes the output of the attention layers.


# Scaled Dot-Product Attention

## Computation

The scaled dot-product attention is a core component of the Transformer architecture. It computes the attention weights as follows:

1. **Query (Q), Key (K), and Value (V):**
   - The input sequence is transformed into three matrices: Q, K, and V, using learnable weight matrices.

2. **Scaled Dot Product:**
   - The dot product of Q and K is computed, and then scaled down by the square root of the dimension of the key vectors (dk).

3. **Softmax:**
   - A softmax function is applied to the scaled dot product to obtain the attention weights.

4. **Weighted Sum:**
   - The attention weights are multiplied with the value matrix V, and the results are summed to produce the output.

## Mathematical Representation

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$


# Summary and Key Takeaways

- Sequence-to-sequence models are powerful tools for mapping input sequences to output sequences.
- Traditional encoder-decoder RNNs suffer from information loss and alignment issues.
- Attention mechanisms address these limitations by allowing the decoder to focus on different parts of the input sequence.
- Self-attention enables the model to learn relationships between words within the same sequence.
- Multi-head attention captures richer contextual information by using multiple attention heads in parallel.
- The Transformer architecture leverages attention mechanisms extensively, achieving state-of-the-art performance in various NLP tasks.


# Review Questions

1. Explain the limitations of traditional encoder-decoder RNNs for sequence-to-sequence tasks.
2. How do attention mechanisms address the limitations of traditional encoder-decoder RNNs?
3. What is the difference between self-attention and cross-attention?
4. Explain the concept of multi-head attention and its benefits.
5. Describe the components of a Scaled Dot-Product Attention head.
6. How does the Transformer architecture differ from traditional seq2seq models based on RNNs?
7. What are some real-world applications of sequence-to-sequence models and attention mechanisms? 
