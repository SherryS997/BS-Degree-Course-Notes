---
title: Decoding Strategies, BERT
---

## Transformer Architecture in Machine Translation

The transformer architecture revolutionized machine translation by replacing recurrent neural networks with an attention-based mechanism. It consists of an encoder and a decoder, both composed of stacked blocks.

### Input Encoding

Both the source and target sequences are first converted into embeddings, which are vectors representing the meaning of words.  Positional encodings are added to these embeddings to provide information about the word order, as the transformer architecture itself doesn't inherently capture sequence order. These encodings are typically sinusoidal functions of the position and dimension:

$$
PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}}) \\
PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

where $pos$ is the position of the word, $i$ is the dimension index, and $d_{model}$ is the embedding dimension.

### Encoder

The encoder consists of $N$ identical layers stacked on top of each other. Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.  A residual connection and layer normalization are applied around each of these two sub-layers.

#### Multi-Head Self-Attention

This mechanism allows the model to attend to different parts of the input sequence when encoding a particular word.  It computes attention weights by projecting the input embeddings into query ($Q$), key ($K$), and value ($V$) matrices. The attention weights are calculated as:

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

where $d_k$ is the dimension of the key vectors. Multi-head attention performs this operation multiple times with different learned projections and concatenates the results.

#### Position-wise Feed-Forward Network

This network consists of two linear transformations with a ReLU activation in between:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$


### Decoder

The decoder also consists of $N$ identical layers.  In addition to the two sub-layers present in the encoder, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, residual connections and layer normalization are applied around each of the sub-layers.

#### Masked Multi-Head Attention

The decoder uses masked multi-head attention to prevent positions from attending to subsequent positions.  This ensures that the prediction for position $i$ depends only on the known outputs at positions less than $i$.

#### Multi-Head Cross-Attention

This mechanism allows the decoder to attend to the encoder's output, effectively incorporating information from the source sequence when generating the target sequence. The query matrix comes from the previous decoder layer, while the key and value matrices come from the encoder output.


### Output Generation

The final decoder layer outputs a vector of logits, which are then passed through a linear layer and a softmax function to produce a probability distribution over the target vocabulary.  The word with the highest probability is chosen as the next word in the translated sequence.

## Transformer Architecture in NLP Tasks

The transformer architecture, initially designed for machine translation, has proven remarkably versatile and effective across a wide range of Natural Language Processing (NLP) tasks.  Instead of training a new architecture for each task, the same underlying transformer structure can be adapted, significantly reducing development time and often leveraging knowledge gained during pre-training on large text corpora.  However, fine-tuning with a task-specific dataset is crucial for optimal performance.

Here's how the transformer is applied to different NLP tasks:

* **Prediction of Class/Sentiment:** In sentiment analysis or other classification tasks, the input text is fed into the transformer. The output is a predicted class label or a sentiment score.  This can be achieved by adding a classification layer on top of the transformer's output representations, typically taking the representation of a special classification token ([CLS]) as input.

* **Text Summarization:** For summarization, the input is the text to be summarized. The transformer generates a condensed version of the input, capturing the key information.  Different approaches exist, including extractive summarization (selecting important phrases from the input) and abstractive summarization (generating new text that summarizes the input).  The transformer can be trained to directly output the summary using sequence-to-sequence learning.

* **Question Answering:** In question answering, the transformer receives both the input text and a question about it.  The model's output is the answer to the question, extracted from or generated based on the input text.  For extractive question answering, the transformer can be trained to predict the start and end positions of the answer span within the input text.  This often involves predicting two probability distributions over the input tokens, one for the start position and one for the end position. For example, given an input sequence of length $n$, the model might predict $s_i$ and $e_j$ representing the probabilities of the $i$-th and $j$-th tokens being the start and end of the answer span, respectively.

$$ s_i = P(\text{start} = i | \text{input text, question}) $$
$$ e_j = P(\text{end} = j | \text{input text, question}) $$

These examples illustrate the adaptability of the transformer architecture. By modifying the inputs and outputs and training on task-specific data, the same core architecture can excel in various NLP tasks.

## Data Challenges and Transformer Models

Labelled data is scarce and expensive to create, posing a significant challenge for training effective NLP models.  Conversely, vast amounts of unlabelled text data are readily available online, presenting an opportunity to improve model performance.  The key challenge lies in how to effectively leverage this unlabelled data. One approach is to use unlabelled data for pre-training a language model and then fine-tune it on a smaller labelled dataset. This helps the model learn general language patterns from the unlabelled data, which can then be refined for specific tasks using the labelled data.

Several questions arise when considering the use of unlabelled data:

* **Training Objective:** What should be the training objective when using unlabelled data?  Traditional supervised learning objectives rely on labelled data.  For unlabelled data, alternative objectives like language modeling, masking, or autoencoding are necessary. These objectives focus on predicting contextual information or reconstructing the input itself, allowing the model to learn inherent language structure.
* **Downstream Task Adaptation:** How can we ensure that the knowledge gained from unlabelled data effectively transfers to downstream tasks?  The goal is to minimize the amount of fine-tuning required on labelled data for each specific task.  Techniques like transfer learning and few-shot learning address this by enabling the model to generalize well from pre-training on unlabelled data to fine-tuning on limited labelled examples.  The success of this adaptation depends on the alignment between the pre-training objective and the downstream tasks.  For example, a model pre-trained on a masking task might adapt better to tasks involving filling missing information, while a model pre-trained on next-word prediction might be more suitable for text generation tasks.
* **Evaluation:** How can we evaluate the effectiveness of pre-training on unlabelled data?  Standard evaluation metrics for supervised tasks require labelled data.  For pre-training, alternative metrics like perplexity (for language models) or reconstruction error (for autoencoders) can be used to assess the model's ability to capture language patterns.  Ultimately, the true test of effective pre-training lies in the performance improvement observed on downstream tasks after fine-tuning.

## Decoding Strategies

Decoding strategies are algorithms used to generate text from language models. They determine how to select the next word in a sequence given the model's predicted probabilities for each word in the vocabulary.  Different strategies offer trade-offs between computational cost, output quality, and diversity.

### Exhaustive Search

Exhaustive search is a decoding strategy that guarantees finding the most probable sequence of words according to the language model. It achieves this by systematically evaluating *every possible* sequence up to a predefined length and selecting the sequence with the highest overall probability.

**Procedure:**

1. **Initialization:** Starting with an empty sequence or a given prompt, the algorithm considers all words in the vocabulary $\mathcal{V}$ as potential candidates for the first word.

2. **Expansion:**  At each subsequent time step $t$, the algorithm expands each existing sequence from the previous step by appending every possible word from the vocabulary. This creates $|\mathcal{V}|$ new sequences for each sequence from the previous step.  If there were $N_{t-1}$ sequences at time step $t-1$, there will be $N_{t-1} \times |\mathcal{V}|$ sequences at time step $t$.

3. **Probability Calculation:** For each newly generated sequence, the algorithm calculates its probability. This probability is the product of the conditional probabilities of each word given the preceding words in the sequence:

   $$ P(w_1, w_2, ..., w_t) = \prod_{i=1}^{t} P(w_i | w_1, w_2, ..., w_{i-1}) $$

   where $w_i$ represents the word at position $i$ in the sequence.  These conditional probabilities are obtained from the language model.

4. **Sequence Selection:**  After generating all possible sequences of the desired length $T$, the algorithm selects the sequence with the highest probability as the output.

**Computational Complexity:**

The main drawback of exhaustive search is its computational cost.  The number of sequences generated grows exponentially with the sequence length.  Specifically, for a vocabulary of size $|\mathcal{V}|$ and a desired sequence length $T$, the algorithm needs to evaluate $|\mathcal{V}|^T$ sequences. This makes exhaustive search impractical for all but the shortest sequences and smallest vocabularies.  For example, with a vocabulary size of 30,000 and a desired sequence length of just 5, the number of sequences to evaluate is $30000^5$, an astronomically large number.

**Advantages:**

* **Optimality:**  Exhaustive search guarantees finding the sequence with the highest probability according to the language model.

**Disadvantages:**

* **Computational Intractability:** The exponential complexity makes it infeasible for practical applications with realistic vocabulary sizes and sequence lengths.

### Greedy Search

Greedy search is a deterministic decoding strategy that selects the word with the highest probability at each time step. This approach is computationally efficient but can lead to suboptimal and repetitive text.

**Algorithm:**

1. **Initialization:** Start with an empty sequence or a given prompt.
2. **Iteration:** For each time step $t$:
   - Obtain the probability distribution $P(w_t | w_{1:t-1})$ over the vocabulary $\mathcal{V}$, conditioned on the previously generated words $w_{1:t-1}$.
   - Select the word $w_t^*$ with the highest probability:
     $$ w_t^* = \arg\max_{w_t \in \mathcal{V}} P(w_t | w_{1:t-1}) $$
   - Append $w_t^*$ to the generated sequence.
3. **Termination:** Stop when a predefined sequence length is reached or a special end-of-sequence token is generated.

**Advantages:**

- **Computational Efficiency:** Greedy search is significantly faster than exhaustive search and less computationally intensive than beam search, as it only requires evaluating $|\mathcal{V}|$ probabilities at each time step.
- **Simplicity:** The algorithm is easy to implement and understand.

**Disadvantages:**

- **Suboptimal Sequences:**  Greedy search may not find the most likely sequence overall. By making locally optimal choices at each time step, it might miss sequences with higher overall probability.  For example, a sequence with a slightly less probable first word could lead to much more probable subsequent words, resulting in a higher overall probability.
- **Lack of Diversity:** Greedy decoding tends to produce repetitive and predictable text.  If the model strongly favors certain words, those words might be repeatedly selected, leading to outputs like "I like to think that I like to think that..."
- **Inability to Recover from Early Mistakes:**  An incorrect word choice early in the generation process can lead to a cascade of errors, as subsequent predictions are conditioned on the erroneous prefix.


**Example:**

Consider a vocabulary $\mathcal{V} = \{\text{the, quick, brown, fox, jumps, over, lazy, dog}\}$.  If the model predicts the following probabilities for the first two words:

- $P(\text{the}) = 0.4$
- $P(\text{quick}) = 0.3$
- $P(\text{brown}) = 0.2$
- $P(\text{fox}) = 0.1$

And for the second word, given the first word is "the":

- $P(\text{quick} | \text{the}) = 0.5$
- $P(\text{brown} | \text{the}) = 0.3$
- $P(\text{fox} | \text{the}) = 0.2$

Greedy search would select "the" as the first word. Then, conditioned on "the", it would select "quick" as the second word. While "the" might have been the most probable first word in isolation, it's possible that another less probable first word could have led to a higher probability second word, making the overall sequence more probable.

### Beam Search

Beam search is a decoding strategy that aims to find a more likely sequence than greedy search while remaining computationally tractable compared to exhaustive search.  It operates by maintaining a set of $k$ most probable sequences (the "beam") at each time step.

The process begins with an initial beam containing the $k$ most likely words for the first position in the sequence.  At each subsequent time step, the algorithm expands each sequence in the beam by considering all possible next words from the vocabulary.  For each expanded sequence, it calculates the probability by multiplying the existing sequence probability by the conditional probability of the new word given the preceding words. This results in $k \times |\mathcal{V}|$ candidate sequences, where $|\mathcal{V}|$ is the vocabulary size.  The algorithm then selects the top $k$ sequences with the highest probabilities from these candidates to form the new beam. This iterative process continues until the desired sequence length is reached.

The final beam contains $k$ complete sequences, and the sequence with the highest overall probability is chosen as the output.  The parameter $k$, called the beam size, controls the trade-off between exploration and computational cost. 

**Illustrative Example:**

Consider a vocabulary $\mathcal{V} = \{A, B, C\}$ and a beam size of $k = 2$. Suppose the model outputs the following conditional probabilities at each time step:

**Time Step 1:**

$P(A) = 0.5$, $P(B) = 0.4$, $P(C) = 0.1$

The initial beam contains the two most likely words: $\{A, B\}$.

**Time Step 2:**

$P(A|A) = 0.1$, $P(B|A) = 0.2$, $P(C|A) = 0.5$

$P(A|B) = 0.2$, $P(B|B) = 0.2$, $P(C|B) = 0.6$

Expanding the beam results in six candidates: $\{AA, AB, AC, BA, BB, BC\}$.  Calculating the probabilities:

$P(AA) = P(A) \times P(A|A) = 0.5 \times 0.1 = 0.05$

$P(AB) = P(A) \times P(B|A) = 0.5 \times 0.2 = 0.1$

$P(AC) = P(A) \times P(C|A) = 0.5 \times 0.5 = 0.25$

$P(BA) = P(B) \times P(A|B) = 0.4 \times 0.2 = 0.08$

$P(BB) = P(B) \times P(B|B) = 0.4 \times 0.2 = 0.08$

$P(BC) = P(B) \times P(C|B) = 0.4 \times 0.6 = 0.24$

The top two sequences with the highest probabilities are $\{AC, BC\}$, which form the new beam.

This process continues for subsequent time steps, with the beam always containing the $k$ most promising sequences.  The final output is the most probable complete sequence in the last beam.

**Strengths and Limitations:**

Beam search offers a balance between finding likely sequences and computational efficiency.  It often produces more fluent and grammatically correct text compared to greedy search.  However, it can still be prone to generating repetitive or predictable outputs, especially when the beam size is small.  The choice of beam size is a crucial factor, influencing the trade-off between diversity and likelihood of the generated text.

### Sampling-based Decoding

Sampling-based methods introduce randomness into the decoding process, allowing for more diverse and potentially creative text generation. These methods don't always choose the most probable word but instead sample from the probability distribution over the vocabulary. This randomness can lead to more interesting and human-like text generation, as it breaks the deterministic nature of greedy and beam search, which tend to produce repetitive and predictable outputs.

#### Temperature Sampling

Temperature sampling modifies the predicted probabilities before sampling, controlling the randomness of the selection. Given logits $u_i$ (pre-softmax output of the model for each word $i$ in the vocabulary) and a temperature parameter $T > 0$, the probabilities are calculated as:

$$ P(x = i|x_{1:t-1}) = \frac{\exp(\frac{u_i}{T})}{\sum_{j} \exp(\frac{u_j}{T})} $$

* **High Temperatures ($T > 1$):**  Flatten the probability distribution, increasing the likelihood of selecting less probable words. This leads to more diverse and surprising outputs, but at the cost of potentially reduced coherence and grammatical correctness.

* **Low Temperatures ($T < 1$):** Concentrate the probability mass on the most likely words, reducing the chance of selecting less probable words. This results in more predictable and grammatically correct outputs, but potentially at the expense of creativity and diversity.

* **Standard Temperature ($T = 1$):** Corresponds to directly sampling from the model's original predicted probabilities, without any modification.

#### Top-K Sampling

Top-k sampling restricts the sampling process to the $k$ most probable words at each time step. The probabilities of these $k$ words are renormalized to sum to 1, and a word is sampled from this modified distribution.

* **Diversity vs. Coherence:** The choice of $k$ determines the trade-off between diversity and coherence in the generated text. Smaller values of $k$ result in more predictable and coherent outputs, as the selection is more focused on the most probable words. Larger values of $k$ allow for more diverse outputs by including less probable words in the sampling pool.

* **Addressing the 'Tail' Problem:** Top-k sampling helps address the issue of sampling from the "tail" of the distribution, where very low probability words might lead to nonsensical or irrelevant outputs. By focusing on the top $k$ words, the sampling process is constrained to more meaningful options.

#### Top-P (Nucleus) Sampling

Top-p sampling, also known as nucleus sampling, dynamically adjusts the number of words considered at each time step based on their cumulative probability. It selects the smallest set of words whose cumulative probability exceeds a predefined threshold $p$ (typically between 0 and 1). The probabilities of these selected words are renormalized, and a word is sampled from this set.

* **Adapting to Probability Distributions:** Top-p sampling adapts to the shape of the probability distribution. When the distribution is flat (high uncertainty), it considers a larger set of words, allowing for more diverse outputs. When the distribution is peaked (high certainty), it focuses on a smaller set of words, resulting in more predictable outputs.

* **Balancing Exploration and Exploitation:** This dynamic selection allows for a balance between exploring less probable words and exploiting the most probable ones, making it a more robust sampling strategy compared to fixed top-k sampling, especially for varying probability distributions.

## Bidirectional Encoder Representations from Transformers (BERT)

BERT leverages the transformer's encoder architecture to learn deep bidirectional representations of text. Unlike unidirectional models like GPT, which process text from left to right, BERT considers the context of both preceding and following words for each token, enabling a richer understanding of language.

### Masked Language Modeling (MLM)

BERT's pre-training relies heavily on Masked Language Modeling (MLM). During pre-training, a portion of the input tokens (typically 15%) is randomly masked. The model then attempts to predict these masked tokens based on the context provided by the surrounding unmasked tokens.  This bidirectional approach allows the model to learn relationships between words in a more comprehensive way compared to unidirectional methods.

### Next Sentence Prediction (NSP)

In addition to MLM, BERT is also pre-trained with Next Sentence Prediction (NSP).  This task involves predicting whether two given sentences are consecutive in the original text.  This helps BERT understand relationships between sentences, beneficial for downstream tasks requiring sentence-level understanding like Question Answering.

### Input Representation

BERT's input representation incorporates three key embeddings for each token:

* **Token Embeddings:**  Represent the individual words in the vocabulary.
* **Segment Embeddings:**  Distinguish between tokens belonging to different segments (sentences). This is crucial for the NSP task.
* **Position Embeddings:**  Encode the position of each token within the sequence.

These three embeddings are summed to create a comprehensive input representation for each token.

### Architecture

The core of BERT is a multi-layer bidirectional transformer encoder. The base model has 12 layers, while the large model has 24 layers. Each layer consists of multi-head self-attention mechanisms and feed-forward networks.  The output of the final encoder layer provides a contextualized representation for each token, capturing the meaning of the word within its context.

### Special Tokens

BERT utilizes special tokens to demarcate segments and handle specific tasks:

* **[CLS]**:  Classification token.  The final hidden representation of this token is typically used for classification tasks.
* **[SEP]**:  Separator token.  Indicates the boundary between sentences.
* **[MASK]**:  Mask token.  Replaces the original token during the MLM task.

### Masking Strategy

BERT's masking strategy is crucial for effective pre-training. The 15% of masked tokens are not simply replaced with [MASK]. Instead:

* 80% are replaced with [MASK].
* 10% are replaced with a random word from the vocabulary.
* 10% remain unchanged.

This approach forces the model to learn more robust representations, as it cannot rely solely on the [MASK] token to identify the missing word.

### Pre-training Data and Objectives

BERT is pre-trained on a massive dataset consisting of BookCorpus (800M words) and English Wikipedia (2.5B words), encompassing a diverse range of topics and writing styles.  The pre-training objective function combines the losses from MLM and NSP:

$$
\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{y_i \in \mathcal{M}} -\log(\hat{y}_i) + \mathcal{L}_{cls}
$$

where:

* $\mathcal{M}$ represents the set of masked tokens.
* $\hat{y}_i$ is the predicted probability distribution for the i-th masked token.
* $\mathcal{L}_{cls}$ is the loss function for the NSP task.

This dual objective allows BERT to learn rich contextualized representations that capture both word-level and sentence-level information, making it highly effective for a wide range of downstream NLP tasks.

### Parameter Calculation for BERT

The BERT base model has approximately 110 million parameters. Let's break down the calculation:

**Embedding Layer:**

* **Token Embeddings:** Vocabulary size ($|V|$) x embedding dimension ($d_{model}$) = 30,522 x 768 ≈ 23.4M parameters.
* **Segment Embeddings:** Number of segments (2) x embedding dimension ($d_{model}$) = 2 x 768 ≈ 1.5K parameters.
* **Position Embeddings:** Maximum sequence length ($T$) x embedding dimension ($d_{model}$) = 512 x 768 ≈ 0.4M parameters.

**Encoder Layers (12 layers in BERT base):**

Each encoder layer has two main components: self-attention and a feed-forward network.

* **Self-Attention:**
    * For each of the 12 attention heads: 
        * Three weight matrices ($W_Q$, $W_K$, $W_V$) for query, key, and value: $d_{model}$ x $d_k$ = 768 x 64 = 49,152 parameters each.
        * One output projection matrix ($W_O$): $d_{model}$ x $d_{model}$ = 768 x 768 = 589,824 parameters.
    * Total parameters per head: 3 x 49,152 + 589,824 ≈ 737,280 parameters.
    * Total parameters for all 12 heads: 12 x 737,280 ≈ 8.8M parameters.
* **Feed-Forward Network:**
    * Two linear transformations with intermediate size 3072:
        * First transformation: $d_{model}$ x 3072 = 768 x 3072 ≈ 2.3M parameters.
        * Second transformation: 3072 x $d_{model}$ = 3072 x 768 ≈ 2.3M parameters.
    * Total parameters for the feed-forward network: 2.3M + 2.3M ≈ 4.6M parameters.

**Total Parameters per Encoder Layer:** 8.8M (self-attention) + 4.6M (FFN) ≈ 13.4M parameters.

**Total Parameters for all 12 Encoder Layers:** 12 x 13.4M ≈ 160.8M parameters.

**Total Parameters in BERT base:**

23.4M (embeddings) + 160.8M (encoders) ≈ **184.2M parameters.**

**Note:** This calculation ignores bias terms and parameters associated with layer normalization for simplicity. The actual number of parameters in the BERT base model is slightly lower, around 110 million, due to parameter sharing within the attention heads and other optimizations.

## Adapting BERT to Downstream Tasks

Pre-trained BERT models can be adapted to various downstream Natural Language Processing (NLP) tasks without extensive modifications.  Two primary approaches are commonly used: feature extraction and fine-tuning.

### Feature Extraction

In feature extraction, BERT acts as a fixed feature encoder.  The input sequence is processed by BERT, and the output representation, typically the hidden state of the [CLS] token or an aggregation of hidden states from the last encoder layer, is used as a feature vector for a separate downstream model.  

This approach is advantageous when labeled data for the downstream task is limited.  Since BERT's parameters are frozen, the risk of overfitting to the small downstream dataset is reduced. However, it may not capture task-specific nuances as effectively as fine-tuning.  

Example: For sentence classification, the feature vector extracted from BERT can be fed into a simple classifier like logistic regression or a support vector machine.

### Fine-tuning

Fine-tuning involves updating BERT's parameters alongside the parameters of a task-specific layer added on top of BERT. The entire model is trained end-to-end on the labeled data for the downstream task.  

This approach allows BERT to adapt to the specific task and potentially achieve better performance than feature extraction.  However, it requires more labeled data and is more susceptible to overfitting if the downstream dataset is small. 

#### Fine-tuning Procedure

1. **Add Task-Specific Layer:** A task-specific layer, such as a classification layer for sentiment analysis or a question answering head for extractive question answering, is added on top of the final BERT encoder layer.  This layer is initialized randomly.

2. **Unfreeze BERT Layers:** While some fine-tuning approaches freeze lower BERT layers to preserve general language knowledge, often, all BERT layers are unfrozen to allow for full adaptation. 

3. **Train on Downstream Data:** The entire model, including BERT and the task-specific layer, is trained on the labeled data for the downstream task.  The loss function is specific to the task (e.g., cross-entropy loss for classification).

4. **Hyperparameter Tuning:** Fine-tuning often requires adjusting hyperparameters such as learning rate, batch size, and the number of training epochs to optimize performance on the downstream task.

**Note:** Masking of input tokens, a core aspect of BERT's pre-training, is typically not performed during fine-tuning. This is because the [MASK] token is not present in downstream tasks, and the model needs to learn to process actual words. 

## Extractive Question Answering with BERT

Extractive Question Answering is a task where, given a question and a context paragraph, the model needs to identify the span of text within the paragraph that answers the question.  BERT can be fine-tuned for this task by adding a prediction layer on top of the pre-trained encoder.

### Input Representation

The question and paragraph are concatenated as input to BERT, separated by the special [SEP] token.  A special [CLS] token is prepended to the beginning of the input.  Each token is embedded using the learned word embeddings, segment embeddings (to distinguish between question and context), and positional embeddings.  This input sequence is then processed by the layers of the BERT encoder.

### Span Prediction

The final hidden representations from the BERT encoder, denoted as  $h_1, h_2, ..., h_n$, where $n$ is the length of the input sequence, are used to predict the start and end positions of the answer span.  Two learnable vectors, $S$ (for start) and $E$ (for end), of the same dimension as the hidden states, are introduced.

The probability of the $i$-th word being the start token is calculated as:

$$ s_i = \frac{\exp(S \cdot h_i)}{\sum_{j=1}^{n} \exp(S \cdot h_j)} $$

Similarly, the probability of the $i$-th word being the end token is:

$$ e_i = \frac{\exp(E \cdot h_i)}{\sum_{j=1}^{n} \exp(E \cdot h_j)} $$

These equations use the dot product between the start/end vectors and the hidden states to capture the relevance of each word to being the start or end of the answer span.  The softmax function normalizes the scores to obtain probabilities.

### Training and Inference

During training, the model is presented with question-paragraph pairs along with the ground-truth start and end positions of the answer span.  The loss function is typically the sum of the cross-entropy losses for the start and end position predictions. This encourages the model to learn the $S$ and $E$ vectors that accurately identify the answer span.

During inference, the model predicts the most likely start and end positions based on the calculated probabilities $s_i$ and $e_i$. The span between these positions is extracted as the answer.  If the predicted end position is before the start position, an empty string is returned, indicating that the model could not find a valid answer span in the context.

## Review Questions

### Transformer Architecture and Applications

1. Explain the key differences between recurrent neural networks and the transformer architecture for sequence processing. What are the advantages of using transformers?
2. Describe the role of positional encodings in the transformer architecture. Why are they necessary? How are they typically calculated?
3. Explain the concept of multi-head attention. What are the benefits of using multiple attention heads?
4. How is the transformer architecture adapted for tasks like sentiment classification, text summarization, and question answering? Provide specific examples for each task.
5. Describe the challenges of using labeled versus unlabeled data in NLP.  How can unlabeled data be leveraged to improve model performance?

### Decoding Strategies

6. What are decoding strategies, and why are they important in text generation?
7. Explain the exhaustive search decoding strategy. Why is it computationally expensive?
8. Describe the greedy search decoding strategy. What are its advantages and disadvantages?
9. How does beam search improve upon greedy search? Explain the role of the beam size in beam search.
10. What are the key differences between temperature sampling, top-k sampling, and top-p sampling?  Explain how each method influences the diversity and coherence of the generated text.
11. When would you prefer one sampling method over the others?  Provide specific scenarios and justifications.

### BERT Architecture and Fine-tuning

12. What does BERT stand for, and what is its main contribution to NLP?  How does it differ from unidirectional language models like GPT?
13. Explain the concept of Masked Language Modeling (MLM) and its role in BERT's pre-training.
14. What is Next Sentence Prediction (NSP), and why is it included in BERT's pre-training objective?
15. Describe the three types of embeddings used in BERT's input representation. Why is each type important?
16. Explain BERT's masking strategy.  Why isn't a simple [MASK] token replacement sufficient for effective pre-training?
17. Describe the two main approaches for adapting BERT to downstream tasks: feature extraction and fine-tuning.  What are the advantages and disadvantages of each approach?  When would you choose one over the other?
18. Explain how BERT can be fine-tuned for extractive question answering.  How are the start and end positions of the answer span predicted?  What loss function is typically used during training?
19. Calculate the approximate number of parameters in a simplified version of the BERT base model.  Break down the calculation by components (embeddings, attention heads, feed-forward networks, etc.). 