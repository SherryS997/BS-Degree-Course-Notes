# Regularization

Regularization techniques play a crucial role in deep learning to address the challenge of overfitting, where the model learns to memorize the training data rather than generalize well to unseen data. In this section, we delve into the concept of regularization, discussing its importance, various techniques, and the trade-off between bias and variance.

## Introduction

In deep learning, models often consist of millions of parameters, while the training data may be limited to only a few million samples. This scenario leads to overparameterization, where the model has more parameters than the available training data points. Consequently, overparameterized models are prone to overfitting, wherein they capture noise and idiosyncrasies in the training data, resulting in poor generalization performance.

## Bias-Variance Trade-off

The bias-variance trade-off is a fundamental concept in machine learning and deep learning. It refers to the balance between bias and variance in the model's predictions.

### Bias
**Bias** represents the error introduced by the model's simplifying assumptions or its inability to capture the underlying structure of the data. In simpler terms, bias measures how much the average prediction of the model deviates from the true value. 

For a regression problem, if the predicted values consistently differ from the actual values, the model is said to have high bias. Conversely, if the model's predictions closely match the true values, it has low bias.

### Variance
**Variance** quantifies the variability of the model's predictions across different training datasets. It measures how much the model's predictions vary for different training samples. 

Models with high variance are sensitive to small fluctuations in the training data, often resulting in overfitting. On the other hand, models with low variance produce consistent predictions across different datasets.

### Trade-off
There exists a trade-off between bias and variance: reducing bias typically increases variance, and vice versa. Finding the right balance between bias and variance is crucial for building models that generalize well to unseen data.

## Example: Fitting a Curve

To illustrate the bias-variance trade-off, consider the task of fitting a curve to a set of data points sampled from a sinusoidal function. We compare two models:

1. **Simple Model**: A linear function $Y = MX + C$
2. **Complex Model**: A degree 25 polynomial

The simple model has low capacity, as it contains only a few parameters, while the complex model has high capacity due to its larger number of parameters.

## Observations

After training the models on different samples of the data, we observe the following:

- **Simple Model**: 
  - Produces similar predictions across different datasets (low variance).
  - However, the average prediction deviates significantly from the true curve (high bias).
- **Complex Model**:
  - Exhibits varied predictions across different datasets (high variance).
  - The average prediction closely matches the true curve (low bias).

These observations highlight the trade-off between bias and variance: simple models tend to underfit the data (high bias, low variance), while complex models tend to overfit (low bias, high variance).

## Formalization

We can formally define bias and variance as follows:

### Bias
The bias of a model is the expected difference between the average prediction of the model and the true value. Mathematically, it can be expressed as:

$$ \text{Bias} = \mathbb{E}[\hat{y}] - y $$

Where:
- $\hat{y}$ represents the average prediction of the model.
- $y$ denotes the true value.

For simple models, the bias tends to be high, indicating a large deviation between the average prediction and the true value. In contrast, complex models exhibit low bias, as their average prediction closely approximates the true value.

### Variance
The variance of a model is the expected squared difference between the model's prediction and its average prediction. Mathematically, it can be defined as:

$$ \text{Variance} = \mathbb{E}[(\hat{y} - \mathbb{E}[\hat{y}])^2] $$

Where:
- $\hat{y}$ represents the model's prediction.
- $\mathbb{E}[\hat{y}]$ denotes the average prediction of the model.

For simple models, the variance tends to be low, indicating consistent predictions across different datasets. In contrast, complex models exhibit high variance, as their predictions vary widely across different datasets.

## Trade-off Revisited

The bias-variance trade-off underscores the need to strike a balance between bias and variance to achieve optimal model performance. Models with excessively high bias may fail to capture the underlying patterns in the data, leading to underfitting. Conversely, models with excessively high variance may capture noise and idiosyncrasies in the training data, leading to overfitting.

# Train Error vs. Test Error

## Introduction

In the realm of deep learning, understanding the behavior of models on both training and test data is crucial for assessing their performance and generalization capabilities. This discussion delves into the concepts of train error versus test error, elucidating their significance in model evaluation and guiding the quest for optimal model complexity.

## Mean Square Error (MSE)

When a deep learning model predicts the output vector $\mathbf{y}$ for a given input vector $\mathbf{x}$, the mean square error (MSE) serves as a metric to quantify the predictive accuracy. Formally, the MSE is computed as the expectation of the squared difference between the predicted and actual outputs:

$$
\text{MSE} = \mathbb{E}[(\hat{\mathbf{y}} - \mathbf{y})^2]
$$

where $\hat{\mathbf{y}}$ represents the network's output, and $\mathbf{y}$ denotes the ground truth output. This expectation captures the average discrepancy between the predicted and true outputs over all possible input-output pairs.

## Dependency on Bias and Variance

The expected error on unseen data is intricately tied to two fundamental properties of a model: bias and variance. 

### Bias

Bias refers to the model's tendency to systematically under- or overestimate the true values. High bias indicates that the model is too simplistic and fails to capture the underlying patterns in the data. In mathematical terms, bias can be represented as:

$$
\text{Bias}(\mathbf{f}) = \mathbb{E}[\hat{\mathbf{y}} - \mathbf{y}]
$$

### Variance

Variance, on the other hand, measures the model's sensitivity to fluctuations in the training data. High variance implies that the model is overly complex and excessively responsive to small variations in the training set. Mathematically, variance can be expressed as:

$$
\text{Var}(\mathbf{f}) = \mathbb{E}[(\hat{\mathbf{y}} - \mathbb{E}[\hat{\mathbf{y}}])^2]
$$

### Trade-off

Achieving low error on unseen data necessitates striking a delicate balance between bias and variance. However, bias and variance are often in tension with each other, making it challenging to simultaneously minimize both. 

## Training and Test Errors

In the context of model evaluation, two pivotal metrics emerge: training error and test error.

### Training Error

The training error quantifies the discrepancy between the model's predictions and the actual outputs on the training data. It serves as a proxy for how well the model fits the training data. Mathematically, the training error is computed as the average squared error over the training set:

$$
\text{Training Error} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

where $N$ is the number of training samples, $\hat{y}_i$ denotes the predicted output for the $i$-th sample, and $y_i$ represents the true output.

### Test Error

In contrast, the test error gauges the model's performance on unseen data that was not used during training. It provides insights into the model's generalization ability. Similar to the training error, the test error is calculated as the average squared error over the test set:

$$
\text{Test Error} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2
$$

where $M$ is the number of test samples, $\hat{y}_i$ denotes the predicted output for the $i$-th test sample, and $y_i$ represents the true output.

## Model Complexity and Error

The relationship between model complexity and error is a central theme in deep learning.

### Impact of Complexity

Increasing the complexity of a model often leads to a reduction in training error. Complex models possess greater capacity to capture intricate patterns in the training data, resulting in improved performance on seen data points.

### Overfitting

However, excessively complex models run the risk of overfitting, wherein they memorize the training data's noise and fail to generalize to new, unseen data. This phenomenon is reflected in an increase in test error despite a decrease in training error.

### Finding the Sweet Spot

The quest for optimal model complexity entails navigating the trade-off between training and test errors. The goal is to identify the "sweet spot" where the model achieves minimal test error without succumbing to overfitting.

## Formal Definitions

Formally defining the training and test errors provides a rigorous framework for model evaluation.

### Training Error

The training error, denoted as $\text{Err}_{\text{train}}$, is computed as the average squared error over the training set:

$$
\text{Err}_{\text{train}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

### Test Error

Similarly, the test error, denoted as $\text{Err}_{\text{test}}$, is calculated as the average squared error over the test set:

$$
\text{Err}_{\text{test}} = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2
$$

## Validation Data

While training error provides insights into the model's performance on seen data points, true evaluation necessitates validation or test data.

### Role of Validation Data

Validation data serves as an independent benchmark for assessing a model's generalization ability. Unlike training data, validation data enables the evaluation of the model's performance on unseen data points, thus offering a more accurate representation of its capabilities.

### Preventing Overfitting

Monitoring test error on validation data facilitates the detection of overfitting. A rise in test error signals the need to curb model complexity to prevent overfitting and ensure robust generalization.

# Estimation and Approximation

## Introduction

In the realm of deep learning, understanding the relationship between data and the underlying true function is crucial for effective modeling. This relationship is often obscured by noise, necessitating approximation techniques to infer the true function. In this discussion, we delve into the process of approximating the true function and estimating the associated mean square error (MSE) to evaluate model performance.

## Data and True Function Relationship

Consider a dataset $D$ comprising both training and test points, where $D$ encompasses $M$ training points and $n$ test points. Within this dataset, there exists a true function $f$ that maps input data $\mathbf{x}$ to output predictions $\mathbf{y}$, subject to some noise $\varepsilon$. Mathematically, this relationship is represented as:

$$
\mathbf{y} = f(\mathbf{x}) + \varepsilon
$$

Here, $\mathbf{y}$ is related to $\mathbf{x}$ via $f$, albeit with added noise $\varepsilon$. We assume $\varepsilon$ follows a zero-centered normal distribution with a small variance $\sigma^2$.

## Approximating the True Function

Since the true function $f$ is unknown, it must be approximated using a surrogate function $\hat{f}$. The parameters of $\hat{f}$ are estimated using the training data $T$, a subset of $D$. Consequently, the prediction of the output becomes:

$$
\mathbf{y} = \hat{f}(\mathbf{x})
$$

By approximating $f$ with $\hat{f}$, we aim to capture the underlying relationship between the input data and the output predictions.

## Mean Square Error (MSE)

Central to assessing model performance is the mean square error (MSE), which quantifies the disparity between predicted and true values. Formally, the MSE is expressed as:

$$
\mathbb{E}[(\hat{f}(\mathbf{x}) - f(\mathbf{x}))^2]
$$

This represents the average squared difference between the predicted value $\hat{f}(\mathbf{x})$ and the true value $f(\mathbf{x})$, computed over numerous samples.

## Estimating the MSE

Directly estimating $\mathbb{E}[(\hat{f}(\mathbf{x}) - f(\mathbf{x}))^2]$ is infeasible due to the unknown true function $f(\mathbf{x})$. Instead, an empirical estimation approach is employed. This involves computing the average square error between predicted and true values using available data. Thus, the empirical estimate substitutes the true expectation with an average computed from observed samples.

## Empirical Estimation Analogies

Empirical estimation of expectations is a common practice across various disciplines. An analogy can be drawn to computing the average number of goals scored in football matches based on a limited set of observed matches. Similarly, in deep learning, the empirical estimate of the MSE is derived from a finite set of training and test data.

## Computing the Empirical Estimate

To compute the empirical estimate of the MSE, the average squared difference between predicted and true values is calculated over the test set. The expected value of $\varepsilon^2$ is $\sigma^2$, representing the variance of the noise.

## Handling Covariance Term

During the derivation of the MSE, a covariance term arises between the noise $\varepsilon$ and the difference between predicted and true values $\hat{f}(\mathbf{x}) - f(\mathbf{x})$. Understanding the influence of this covariance term is essential for accurate estimation of the MSE.

## Independence of Covariance Term

The noise $\varepsilon$ is independent of the difference $\hat{f}(\mathbf{x}) - f(\mathbf{x})$ since the test data used to compute $\varepsilon$ does not participate in the training of $\hat{f}(\mathbf{x})$. Consequently, the covariance between $\varepsilon$ and $\hat{f}(\mathbf{x}) - f(\mathbf{x})$ is zero.

## Impact on Estimation

When estimating the MSE from test data, the covariance term becomes zero, simplifying the estimation process. Thus, the true error is closely approximated by the empirical test error plus a small constant ($\sigma^2$).

## Avoiding Bias in Estimation

Estimating model performance solely from training data yields overly optimistic results. To obtain a more accurate assessment, the test error, which reflects the true error, should be employed. By empirically estimating the error from test data, a realistic depiction of model performance can be attained.