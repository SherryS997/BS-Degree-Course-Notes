---
title: Convolutional Neural Networks
---

# Introduction

Convolutional Neural Networks (CNNs) are a type of neural network architecture that have revolutionized the field of image and signal processing. They are designed to take advantage of the spatial structure in data, such as images, and have been instrumental in achieving state-of-the-art performance in various computer vision tasks.

Convolution Operation
-------------------------

The convolution operation is a fundamental component of CNNs. It is a weighted sum of the input data, where the weights are learned during training. Mathematically, the convolution operation can be represented as:

$$\mathbf{y} = \mathbf{W} \ast \mathbf{x} + \mathbf{b}$$

where $\mathbf{x}$ is the input vector, $\mathbf{W}$ is the weight matrix, $\mathbf{b}$ is the bias vector, and $\ast$ denotes the convolution operator.

1D Convolution
-----------------

In 1D convolution, a filter (also known as a kernel) is applied to a 1D input data. The filter slides over the input data, and at each position, the dot product of the filter and the input data is computed. The resulting output is a feature map, which represents the presence of a particular feature in the input data.

Mathematically, the 1D convolution operation can be represented as:

$$y[i] = \sum_{m=0}^{M-1} x[i+m] \cdot w[m] + b$$

where $y[i]$ is the output at position $i$, $x[i]$ is the input data at position $i$, $w[m]$ is the filter weight at position $m$, and $b$ is the bias term.

2D Convolution
-----------------

In 2D convolution, a 2D filter is applied to a 2D input data, such as an image. The filter slides over the input data, and at each position, the dot product of the filter and the input data is computed. The resulting output is a feature map, which represents the presence of a particular feature in the input data.

Mathematically, the 2D convolution operation can be represented as:

$$y[i, j] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} x[i+m, j+n] \cdot w[m, n] + b$$

where $y[i, j]$ is the output at position $(i, j)$, $x[i, j]$ is the input data at position $(i, j)$, $w[m, n]$ is the filter weight at position $(m, n)$, and $b$ is the bias term.

Centered Formula for 2D Convolution
-------------------------------------

The centered formula for 2D convolution is:

$$y[i, j] = \sum_{m=-M/2}^{M/2} \sum_{n=-N/2}^{N/2} x[i+m, j+n] \cdot w[m, n] + b$$

where $M$ and $N$ are the dimensions of the filter.

Examples of 2D Convolution
-----------------------------

* Blurring: A filter with all weights equal to $1/9$ can be used to blur an image.
* Sharpening: A filter with a weight of $5$ at the center and $-1$ at the surrounding pixels can be used to sharpen an image.
* Edge Detection: A filter with weights of $-1, -1, -1, -1, 8, -1, -1, -1, -1$ can be used to detect edges in an image.

3D Convolution
-----------------

In 3D convolution, a 3D filter is applied to a 3D input data, such as a color image. The filter slides over the input data, and at each position, the dot product of the filter and the input data is computed. The resulting output is a feature map, which represents the presence of a particular feature in the input data.

Mathematically, the 3D convolution operation can be represented as:

$$y[i, j, k] = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} \sum_{o=0}^{O-1} x[i+m, j+n, k+o] \cdot w[m, n, o] + b$$

where $y[i, j, k]$ is the output at position $(i, j, k)$, $x[i, j, k]$ is the input data at position $(i, j, k)$, $w[m, n, o]$ is the filter weight at position $(m, n, o)$, and $b$ is the bias term.

Relationship between Input Size, Output Size, and Filter Size
----------------------------------------------------------------

The output size of a convolution operation depends on the input size, filter size, and stride. The formula for the output size is:

$$W_2 = \frac{W_1 - F + 2P}{S} + 1$$

$$H_2 = \frac{H_1 - F + 2P}{S} + 1$$

where $W_1$ and $H_1$ are the input sizes, $F$ is the filter size, $P$ is the padding, and $S$ is the stride.

Padding
---------

Padding is used to ensure that the output size is the same as the input size. The padding is added to the input data, and the filter is applied to the padded input data.

Stride
---------

The stride defines the interval at which the filter is applied. A stride of $2$ means that the filter is applied to every other pixel, resulting in an output size that is half of the input size.

Depth of the Output
---------------------

The depth of the output is equal to the number of filters used in the convolution operation. Each filter produces a 2D feature map, and the depth of the output is the number of feature maps.

