# Explore the variants of the ReLU activation function in multi-layer networks trained on MNIST dataset
In this small project, I investigate the performance of different activation functions in multi-layer networks to address the MNIST digit classification problem. I compare the performances of the five activation functions (Sigmoid, ReLU, Leaky ReLU, ELU and SELU) on the same database. I find that ReLU and its further functions have better performance than Sigmoid function and the variants of the ReLU activation function are better than ReLU. Then I choose one activation function to explore the impact of the depth of the network. After comparing the results of different numbers of hidden layers, I can see that the network with five hidden layers has the best performance in this specific case. In addition, I compare the different weight initialization strategies and in the experiment the Uniform distribution performs better than Gaussian distribution.

## Introduction
In recent years, the deep learning in the field of computer vision has made remarkable achievements, one of the important factors is the development activation function. As I know, there are many papers focusing on exploring the best activation function. I think it is meaningful to have some conclusions by myself from some experiments. Thus I need to answer the following questions from my experiments. The first one is the performance of different activation functions. Secondly, I need to explore the impact of the depth of the hidden layer to the network. Finally, I need to investigate different approaches to weight initialization. In all my experiments, the batch size is 50 and train for 100 epochs.

## Activation functions
In this section, I mainly introduce the four activation functions: Rectified Linear Unit (ReLU), Leaky Rectified Linear Unit (Leaky ReLU), Exponential Linear Unit (ELU) and Scaled Exponential Linear Unit (SELU). In the following subsections, I use equations to introduce each activation function mathematically.

#### Rectified Linear Unit
Formally, the ReLU is defined as:
