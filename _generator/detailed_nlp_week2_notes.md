## NLP Pipeline

### Steps in the NLP Pipeline

1. **Data Collection**
   - Gathering the initial dataset for processing.

2. **Text Cleaning**
   - Removing noise and irrelevant information from the text.
   - Handling missing or inconsistent data.

3. **Pre-processing**
   - Converting text into a standardized format.
   - Tokenization and normalization.

4. **Feature Engineering**
   - Extracting relevant features from the pre-processed data.
   - Creating numerical representations of text.

5. **Modeling**
   - Building and training the NLP models.
   - Utilizing algorithms like SVM, LSTM, or transformers.

6. **Evaluation**
   - Assessing the performance of the models.
   - Using metrics such as accuracy, precision, recall, and F1-score.

7. **Deployment**
   - Implementing the trained model in a production environment.
   - Ensuring scalability and efficiency.

8. **Monitoring and Model Updating**
   - Continuously monitoring the performance of the deployed model.
   - Updating the model with new data and improvements. 


## Data Collection

### Challenges:

#### Data Accessibility

- Limited access to proprietary or sensitive data
- High costs of acquiring certain datasets
- Technical barriers to accessing data from various sources

#### Data Volume and Variety

- Managing large volumes of data (Big Data)
- Integrating data from multiple sources and formats
- Handling unstructured data (e.g., text, images, videos)

#### Bias and Representativeness

- Ensuring the data is representative of the population
- Avoiding sampling bias
- Addressing any inherent biases in the data collection process 


## Text Cleaning

### Remove Noise

#### Punctuation, Numbers, and Special Characters

- **Original Text:** "Hello! This is an example text with numbers 12345 and symbols $%&."
- **Cleaned Text:** "Hello This is an example text with numbers and symbols"
- Removing noise helps focus on the meaningful parts of the text.

### Correct Spelling Errors and Normalize Text

- **Original Text:** "This sentence contains a spelling error."
- **Corrected Text:** "This sentence contains a spelling error."
- Normalization involves converting text to a standard form, such as converting different forms of a word to a single form (e.g., "color" and "colour" to "color").

### Handle Misspellings, Slang, and Abbreviations

- **Original Text:** "OMG, this txt is gr8!"
- **Normalized Text:** "Oh my god, this text is great!"
- Converting slang and abbreviations to their full forms ensures clarity and consistency. 


## Text Preprocessing Steps

### Tokenization

- Split text into individual words or sentences.
- Example: "This is an example sentence." -> ["This", "is", "an", "example", "sentence."]

### Lowercasing

- Convert all text to lowercase to ensure consistency.
- Example: "This Is An Example Sentence." -> "this is an example sentence."

### Stop Words Removal

- Eliminate common words (e.g., "and", "the") that add little value.
- These words are often high-frequency but do not contribute significantly to the meaning of the text.

### Normalization

- Convert text into a standardized format by addressing various inconsistencies and variations.
- Examples:
    - Convert different forms of a word to a single form (e.g., "color" and "colour" to "color").
    - Replace contractions with their full forms (e.g., "don't" to "do not").

### Stemming/Lemmatization

- Reduce words to their base or root form.
- Stemming: A simpler approach that often results in non-dictionary words (e.g., "running" -> "run").
- Lemmatization: A more complex approach that produces actual dictionary words (e.g., "running" -> "run"). 


## Stemming and Lemmatization

### Stemming

Stemming is a process that removes suffixes from words to reduce them to a base form. It uses heuristic rules and does not always produce valid dictionary words.

**Example:**

- **Original Word:** Running
- **Stemmed Word:** Run

### Lemmatization

Lemmatization reduces words to their base or dictionary form (lemma) by considering the context and ensuring the root form is a valid word. It involves more complex analysis compared to stemming.

**Example:**

- **Original Word:** Running
- **Lemmatized Word:** Run

**Note:**  The provided example uses the same word for both stemming and lemmatization. However, in many cases, these processes can produce different results depending on the specific word and the algorithm used. 


## Text Analysis: Stemming and Lemmatization

### Stemming

Stemming is a process that removes suffixes from words to reduce them to a base form. It uses heuristic rules and does not always produce valid dictionary words.

**Example:**

- **Original Word:** Running
- **Stemmed Word:** Run

### Lemmatization

Lemmatization reduces words to their base or dictionary form (lemma) by considering the context and ensuring the root form is a valid word. It involves more complex analysis compared to stemming.

**Example:**

- **Original Word:** Running
- **Lemmatized Word:** Run

**Note:**  The provided example uses the same word for both stemming and lemmatization. However, in many cases, these processes can produce different results depending on the specific word and the algorithm used. 


## Text Analysis: Linguistic Representation

### Tokenization

Tokenization is the process of splitting text into individual units called tokens. These tokens can be words, phrases, or other meaningful segments. For example:

- **Input Text**: "Text analysis is crucial for understanding linguistic representation."
- **Tokens**: ["Text", "analysis", "is", "crucial", "for", "understanding", "linguistic", "representation"]

### Parsing

Parsing involves analyzing the grammatical structure of the sentence. This typically includes:

- Identifying the subject and predicate.
- Recognizing verbs, nouns, and other parts of speech.
- Determining the syntactic relationships between words.

For example, the sentence "Text analysis is crucial for understanding linguistic representation" can be parsed as follows:

- **Subject**: "Text analysis"
- **Predicate**: "is crucial for understanding linguistic representation"

### Part-of-Speech Tagging

Part-of-speech tagging involves assigning a part of speech (e.g., noun, verb, adjective) to each word in the text. For example:

```plaintext
Text/NNP analysis/NN is/VBZ crucial/JJ for/IN understanding/VBG linguistic/JJ representation/NN .
```

### Named Entity Recognition

Named entity recognition (NER) involves identifying and categorizing named entities in the text. Common categories include:

- **Person (PER)**: Names of people.
- **Organization (ORG)**: Names of organizations.
- **Location (LOC)**: Names of locations.
- **Date (DATE)**: Specific dates.
- **Time (TIME)**: Specific times.

For example, the sentence "Steve Jobs founded Apple Inc. in Cupertino, California" can be tagged as follows:

```plaintext
Steve/NNP Jobs/NNP founded/VBD Apple/NNP Inc./ORG in/IN Cupertino/NNP ,/ , California/NNP .
```

### Sentiment Analysis

Sentiment analysis determines the emotional tone or sentiment expressed in the text. It can be categorized as:

- **Positive**: Expressing a positive sentiment.
- **Negative**: Expressing a negative sentiment.
- **Neutral**: Expressing no particular sentiment.

For example, the sentence "This product is fantastic!" expresses a positive sentiment, while "This is the worst product I have ever seen" expresses a negative sentiment. 


## Advantages and Disadvantages of Stemming

### Advantages of Stemming

- Fast and simple to implement.
- Reduces dimensionality of text data, making it easier to analyze.

### Disadvantages of Stemming

- Sometimes too aggressive, leading to non-words.
    - Example: "studies" -> "studi"
- May result in words that lose their meaning.
    - Example: "caring" -> "car" 


## Practical Applications of Lemmatization

### Machine Translation

- Ensures that words are translated accurately by maintaining their base form.

### Sentiment Analysis

- Improves the accuracy of text sentiment analysis by understanding the correct form of words.

### Speech Recognition

- Helps in identifying the correct form of spoken words to improve transcription accuracy. 


## Complex Morpheme Segmentation in Some Languages

### Turkish

- Some languages, like Turkish, require complex morpheme segmentation.
- **Example:** Uygarlastiramadiklarimizdanmissinizcasina
  - This word translates to: '(behaving) as if you are among those whom we could not civilize'
  - Segmentation: 
    - Uyg'ar 'civilized' + las 'become' + tir 'cause' + ama 'not able' + dik 'past' + lar 'plural' + imiz '1pl' + dan 'ablative' + mis 'past' + siniz '2pl' + casina 'as if'
- This example highlights the complex nature of morpheme segmentation in Turkish, where a single word can consist of multiple morphemes that contribute to its overall meaning. 


## Morphological Typology

### Isolating

- **Example:** Mandarin
    - **méi** - America
    - **guó** - country
    - **rén** - person
    - **méi guó rén** - American

- **Characteristics:**
    - Words are typically monosyllabic and consist of a single morpheme.
    - Morphemes are rarely bound.
    - Grammatical relations are often expressed through word order or particles.


### Agglutinative

- **Example:** Tamil
    - **pe.su** - speak
    - **kir** - present
    - **e.n** - 1st person singular
    - **pe.su kir e.n** - I am speaking

- **Characteristics:**
    - Words can have multiple suffixes attached to a root.
    - Suffixes are typically bound morphemes.
    - Grammatical relations are often expressed through a series of suffixes.


### Fusional

- **Example:** Spanish
    - **ind** - present indicative
    - **hablar** - speak
    - **yo** - 1st person singular
    - **yo hablar** - I speak

- **Characteristics:**
    - Morphemes are often fused together, resulting in complex forms.
    - Grammatical relations are often expressed through inflectional endings that combine multiple meanings.


### Polysynthetic

- **Example:** Mohawk
    - **s** - again
    - **a** - past
    - **hiywa** - she/him
    - **nho** - door
    - **tu** - close
    - **kw** - un
    - **eha** - for
    - **e'r** - perfect
    - **s a hiywa nho tu kw eha e'r** - she opened the door for him again

- **Characteristics:**
    - Words can be extremely long and complex, consisting of multiple morphemes.
    - Morphemes are often bound and express multiple grammatical relations within a single word.
    - Often found in languages spoken by small groups of people who rely heavily on verbal communication.


## Computational Morphology

### What is Computational Morphology?

- Computational morphology focuses on developing techniques and theories for analyzing and synthesizing word forms using computers.

### What do you need to understand?

- **Theoretical knowledge of morphology of languages:** Understanding the structure and formation of words in different languages.
- **Computational techniques for implementation:** Using algorithms and programming to implement these morphological analyses.

### Where is the application?

- **Hyphenation:** Breaking up words into syllables for better readability.
- **Spell Checking:** Identifying and correcting misspelled words.
- **Stemmers:** Reducing words to their base form, often used in information retrieval.
- **Machine Translation:** Translating text accurately by handling word forms across languages.
- **QA system:**  Answering questions based on text by understanding word structure.
- **Content Analysis:**  Analyzing textual content by understanding the meaning of words.
- **Speech Synthesis:**  Generating artificial speech by synthesizing word sounds. 


## Concepts of Morphology

### Null Hypothesis

- Morphological processing might be unnecessary since every word in a language could be stored and accessed directly.

### Morphological Rules

- The number of possible words in a language is infinite.
- The number of actual words is also very large.
- Therefore, we need **morphological rules** or **word formation strategies** to recognize and produce new words. 


## Two Basic Divisions in Morphology

### Inflectional Morphology (Conjugation/Declension)

- This branch deals with changes in word forms that express grammatical functions like tense, number, gender, case, and mood.
- Examples:
    - Verb conjugation: "walk" -> "walked", "walking", "walks"
    - Noun declension: "cat" -> "cats", "cat's"

### Lexical Morphology (Word Formation)

- This branch focuses on the creation of new words through various processes, including:
    - **Derivation:** Adding prefixes or suffixes to existing words (e.g., "un-happy", "happy-ness")
    - **Compounding:** Joining two or more words (e.g., "sunrise", "blackboard")
    - **Conversion:** Shifting a word from one part of speech to another (e.g., "google" [verb] from "Google" [noun])
    - **Acronymy:**  Creating a word from the initial letters of a phrase (e.g., "NATO", "laser")
    - **Clipping:** Shortening a word (e.g., "photo" from "photograph")
    - **Blending:** Combining parts of two words (e.g., "smog" from "smoke" and "fog") 


## Building Blocks of Morphology

### Morpheme

**Morpheme**: the smallest meaningful linguistic unit. Some morphemes are identical with words, but many morphemes are smaller than words.

#### Morpheme ≤ Word

- **morphemes**
  - **free**
    - lexical (child, teach)
    - functional (and, the)
  - **bound**
    - derivational (re-, -ness)
    - inflectional (-'s, -ed)

![Diagram](image_url_placeholder) - [Interpreted: A diagram depicting the relationship between morphemes and words, likely showing different types of morphemes and their relation to free and bound morphemes.] 


## Free Morpheme

A **free morpheme** is a morpheme that can stand alone, that is a complete word; an independent morpheme.

**Example**:
- *walk*, *book*, *but*, *of* and etc.,

## Lexical Morpheme

A **lexical morpheme** is a morpheme that denotes the content words. They **receive inflection**.

- **Open class categories**:
  - Nouns
  - Verbs
  - Adjectives

- **Closed class categories**:
  - Pronouns
  - Number words
  - Nouns of space and time (NST)

## Functional Morpheme

A **functional morpheme** is a morpheme that denotes the functional words. They **do NOT receive inflection** and are **indeclinables** or **avyayas**.

- **Categories**:
  - Prepositions/Postpositions
  - Conjunctions
  - Interjections
  - Adverbs
  - Demonstratives
  - Intensifiers
  - Quotatives etc., 

[Interpreted: This section likely contained a diagram showing the relationship between different types of morphemes. However, without the actual image, we can only describe the content in text form.] 


## Non-concatenative phenomena

### Root-template morphology

- Some languages, like Semitic languages, exhibit a type of morphology known as *root-template morphology*.
- This morphology involves a root consisting of consonants, which is combined with different vowel patterns (templates) to create words with related meanings.

### Example: Arabic root *"ktb"*

- The Arabic root *"ktb"* represents the concept of "writing".
- By combining this root with different templates, we can generate various words related to writing:

| Template          | a (active)      | u (passive)      | Meaning               |
|-------------------|------------------|------------------|------------------------|
| **CVCVC**         | katab           | kutib           | 'write'               |
| **CVCCVC**        | kattab          | kuttib          | 'cause to write'      |
| **CVVCVC**        | ka:tab          | ku:tab          | 'correspond'          |
| **tVCVVCVC**      | taka:tab        | tuku:tab        | 'write each other'    |
| **nCVCVVCVC**     | nka:tab         | nku:tab         | 'subscribe'           |
| **CtVCVC**        | ktatab          | ktutib          | 'write'               |
| **stVCCVC**       | staktab         | stukib          | 'dictate'             |

- This example demonstrates how a single root can be used to create numerous related words by changing the vowel pattern, effectively changing the meaning without adding new morphemes. 


## Example from Telugu (Dravidian)

- illu 'house'
- iMti- ni 'house (object)'
- iMti- ki 'to the house'
- iMti- lō 'in the house'
- iMti- tō 'with the house'
  ...
  
## rA 'to come'

- vacc- A- nu 'I came'
- vACC- A- mu 'we came'
- vACC- A- vu 'you(sg.) came'
- vACC- A- ru 'you(pl.) came'
- vACC- A- du 'he came'


## Incorporating Languages (polysynthetic)

### Characteristics

- **All bound forms are affixes:** All bound forms in polysynthetic languages are affixes, attached to the base of a word to form new words.
- **Inflections are incorporated into the word:** Inflections, which indicate grammatical categories like tense, number, or case, are incorporated directly into the word itself.
- **Ability to form words that are equivalent to whole sentences in other languages:** These languages can create words that convey the same meaning as entire sentences in other languages.
- **Morphologically extremely complex:** Polysynthetic languages have extremely complex morphology, involving a high degree of word formation using affixes.
- **Generally, morphology is more important than context and syntax:** Morphology plays a more critical role than context and syntax in determining the meaning and structure of sentences in these languages.

### Examples

- Icelandic
- Aleutian 


## Incorporating Languages (polysynthetic)

### Characteristics

- **All bound forms are affixes:** All bound forms in polysynthetic languages are affixes, attached to the base of a word to form new words.
- **Inflections are incorporated into the word:** Inflections, which indicate grammatical categories like tense, number, or case, are incorporated directly into the word itself.
- **Ability to form words that are equivalent to whole sentences in other languages:** These languages can create words that convey the same meaning as entire sentences in other languages.
- **Morphologically extremely complex:** Polysynthetic languages have extremely complex morphology, involving a high degree of word formation using affixes.
- **Generally, morphology is more important than context and syntax:** Morphology plays a more critical role than context and syntax in determining the meaning and structure of sentences in these languages.

### Examples

- Icelandic
- Aleutian 

## Computational Morphology

### What is Computational Morphology?

- Computational morphology focuses on developing techniques and theories for analyzing and synthesizing word forms using computers.

### What do you need to understand?

- **Theoretical knowledge of morphology of languages:** Understanding the structure and formation of words in different languages.
- **Computational techniques for implementation:** Using algorithms and programming to implement these morphological analyses.

### Where is the application?

- **Hyphenation:** Breaking up words into syllables for better readability.
- **Spell Checking:** Identifying and correcting misspelled words.
- **Stemmers:** Reducing words to their base form, often used in information retrieval.
- **Machine Translation:** Translating text accurately by handling word forms across languages.
- **QA system:**  Answering questions based on text by understanding word structure.
- **Content Analysis:**  Analyzing textual content by understanding the meaning of words.
- **Speech Synthesis:**  Generating artificial speech by synthesizing word sounds. 


## Morphological Modeling

### Models of Morphological Formations

Morphologists propose three models to describe how words are formed:

1. **Item and Arrangement (IA)**
   - This model views words as being composed of a set of basic units (morphemes) that are arranged in a specific order.
   - The meaning of a word is determined by the combination of morphemes and their order.
   - **Example:** "un-happy-ness" is formed by combining the morphemes "un-", "happy", and "-ness" in that order.

2. **Item and Process (IP)**
   - This model considers words as being formed by a process of applying rules to a set of basic units (morphemes).
   - The rules can involve adding, deleting, or modifying morphemes.
   - **Example:** The plural form of "cat" is formed by adding the "-s" morpheme to the base form "cat".

3. **Word and Paradigm (WP)**
   - This model sees words as being part of a paradigm, a set of related forms that share a common base form.
   - The meaning of a word is determined by its position within the paradigm.
   - **Example:** The verb "walk" has a paradigm that includes forms like "walks", "walked", "walking", etc. The meaning of each form is determined by its position within the paradigm. 

[Interpreted: This diagram is likely a visual representation of the three models of morphological formations, showing their key features and differences.] 


## Computational model: Finite State Technology

### Finite State Automata (FSA)

**Finite State Automata (FSA)** is a abstract mathematical device which describes processes involving inputs and processing it. FSA may have several states and switches between them. Each state is crossed depending on the input symbol and performs the computational tasks associated with the input. A Finite State Automaton is a machine composed of

- An input tape
- A finite number of states, with one initial and one or more accepting states
- Actions in terms of transitions from one state to the other, depending on the current state and the input

[Interpreted: A diagram of Finite State Automata is mentioned, but not provided. It would likely show a machine with states represented as circles, transitions between states as arrows labeled with input symbols, and a starting state and accepting states marked.]

### Finite State Transducers

[Interpreted: This topic is mentioned but not explained. It is likely a type of FSA that outputs a value for each input, but this needs further clarification from the OCR text.] 


## Finite State Transducer:

- **FST**, unlike **FSA**, works on two tapes: input and output tape.
- **FSAs** can recognize a string but do not give the internal structures.
- **But FSTs** can recognize and are able to provide the internal structure of any input.
- They read from one tape and write on another tape.
- So it is possible to turn **FST** to analyze and generate the forms.

[Interpreted: A diagram of a Finite State Transducer is mentioned, but not provided. It would likely show a machine with two tapes, one for input and one for output, states represented as circles, transitions between states as arrows labeled with input and output symbols, and a starting state and accepting states marked.] 


## Morphological Analyzers & Generators

### Morphological Analysis

#### Stemming

- The process of reducing a word to its base or root form.
  - Example: "running" -> "run"

#### Lemmatization

- The process of converting a word to its base or dictionary form.
  - Example: "better" -> "good"

### Morphological Generation

#### Inflection

- The process of changing a word to indicate grammatical information such as tense, number, or case.
  - Example: "walk" -> "walks"

#### Derivation

- The process of forming new words by adding affixes to the base word.
  - Example: "happy" -> "happiness"

### Applications

- **Natural Language Processing (NLP)**: Morphological analysis and generation are crucial in NLP tasks such as tokenization, part-of-speech tagging, and named entity recognition.
- **Machine Translation**: These techniques help in translating words accurately between different languages.
- **Spelling Correction**: Morphological rules can be used to suggest correct spellings and word completions.

### Algorithms and Tools

- **Stemmers**: Tools that perform stemming, such as Porter Stemmer and Snowball Stemmer.
- **Lemmatizers**: Tools that perform lemmatization, such as WordNet Lemmatizer.
- **Morphological Generators**: Tools that generate inflections and derivations using rule-based systems.

### Example

Consider the word "unhappiness":

- **Analysis**:
  - Base form: "happy"
  - Negation prefix: "un-"
  - Noun suffix: "-ness"

- **Generation**:
  - Combine the morphemes: "un-" + "happy" + "-ness" -> "unhappiness"


