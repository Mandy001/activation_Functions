# Explore the variants of the ReLU activation function in multi-layer networks trained on MNIST dataset
In this small project, I investigate the performance of different activation functions in multi-layer networks to address the MNIST digit classification problem. I compare the performances of the five activation functions (Sigmoid, ReLU, Leaky ReLU, ELU and SELU) on the same database. I find that ReLU and its further functions have better performance than Sigmoid function and the variants of the ReLU activation function are better than ReLU. Then I choose one activation function to explore the impact of the depth of the network. After comparing the results of different numbers of hidden layers, I can see that the network with five hidden layers has the best performance in this specific case. In addition, I compare the different weight initialization strategies and in the experiment the Uniform distribution performs better than Gaussian distribution.

## Introduction
In recent years, deep learning in the field of computer vision has made remarkable achievements, one of the important factors is the development of activation functions. As I know, there are many papers focusing on exploring the best activation function. I think it is meaningful to have some conclusions by myself from some experiments. Thus I need to answer the following questions from my experiments. The first one is the performance of different activation functions. Secondly, I need to explore the impact of the depth of the hidden layer to the network. Finally, I need to investigate different approaches to weight initialization. In all my experiments, the batch size is 50 and train for 100 epochs.

## Activation functions
In this section, I mainly introduce the four activation functions: Rectified Linear Unit (ReLU), Leaky Rectified Linear Unit (Leaky ReLU), Exponential Linear Unit (ELU) and Scaled Exponential Linear Unit (SELU). In the following subsections, I use equations to introduce each activation function mathematically.
#### Rectified Linear Unit
Formally, the ReLU is defined as:

  <a href="http://www.codecogs.com/eqnedit.php?latex=relu(x)&space;=&space;max(0,&space;x)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?relu(x)&space;=&space;max(0,&space;x)" title="relu(x) = max(0, x)" /></a>
  which has the gradient:
  
<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}relu(x)=\begin{cases}&space;0&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}relu(x)=\begin{cases}&space;0&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" title="\frac{\mathrm{d} }{\mathrm{d} x}relu(x)=\begin{cases} 0 & \text{ if } x\leq 0 \\ 1 & \text{ if } x> 0 \end{cases}" /></a>

#### Leaky Rectified Linear Unit
The Leaky ReLU has the following form:

<a href="http://www.codecogs.com/eqnedit.php?latex=lrelu(x)=\begin{cases}&space;\alpha&space;x&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;x&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?lrelu(x)=\begin{cases}&space;\alpha&space;x&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;x&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" title="lrelu(x)=\begin{cases} \alpha x & \text{ if } x\leq 0 \\ x & \text{ if } x> 0 \end{cases}" /></a>

which has the gradient:

<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}lrelu(x)=\begin{cases}&space;\alpha&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}lrelu(x)=\begin{cases}&space;\alpha&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" title="\frac{\mathrm{d} }{\mathrm{d} x}lrelu(x)=\begin{cases} \alpha & \text{ if } x\leq 0 \\ 1 & \text{ if } x> 0 \end{cases}" /></a>

typically <a href="http://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> = 0.01 and in this experiment <a href="http://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> = 0.01

#### Exponential Linear Unit
The equation of the ELU is:

<a href="http://www.codecogs.com/eqnedit.php?latex=elu(x)=\begin{cases}&space;\alpha(exp(x)-1)&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;x&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?elu(x)=\begin{cases}&space;\alpha(exp(x)-1)&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;x&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" title="elu(x)=\begin{cases} \alpha(exp(x)-1) & \text{ if } x\leq 0 \\ x & \text{ if } x> 0 \end{cases}" /></a>

which has the gradient:

<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}elu(x)=\begin{cases}&space;\alpha&space;exp(x)&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}elu(x)=\begin{cases}&space;\alpha&space;exp(x)&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" title="\frac{\mathrm{d} }{\mathrm{d} x}elu(x)=\begin{cases} \alpha exp(x) & \text{ if } x\leq 0 \\ 1 & \text{ if } x> 0 \end{cases}" /></a>

typically <a href="http://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> = 1 and in this experiment <a href="http://www.codecogs.com/eqnedit.php?latex=\alpha" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /></a> = 1

#### Scaled Exponential Linear Unit
Mathematically, the SELU is defined as:

<a href="http://www.codecogs.com/eqnedit.php?latex=selu(x)=\lambda&space;\begin{cases}&space;\alpha(exp(x)-1)&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;x&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?selu(x)=\lambda&space;\begin{cases}&space;\alpha(exp(x)-1)&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;x&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" title="selu(x)=\lambda \begin{cases} \alpha(exp(x)-1) & \text{ if } x\leq 0 \\ x & \text{ if } x> 0 \end{cases}" /></a>

which has the gradient:

<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}selu(x)=\lambda&space;\begin{cases}&space;\alpha&space;exp(x)&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\mathrm{d}&space;}{\mathrm{d}&space;x}selu(x)=\lambda&space;\begin{cases}&space;\alpha&space;exp(x)&space;&&space;\text{&space;if&space;}&space;x\leq&space;0&space;\\&space;1&space;&&space;\text{&space;if&space;}&space;x>&space;0&space;\end{cases}" title="\frac{\mathrm{d} }{\mathrm{d} x}selu(x)=\lambda \begin{cases} \alpha exp(x) & \text{ if } x\leq 0 \\ 1 & \text{ if } x> 0 \end{cases}" /></a>

in this experiment, <a href="http://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> = 1.6733 and <a href="http://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> = 1.0507

## Experimental comparison of activation functions

In this section, I use five activation functions (Sigmoid, ReLU, Leaky ReLU, ELU and SELU) to carry out the experiments on the MNIST task. I use two hidden layers with 100 hidden units per layer for each experiment. The results are showed in Figure~\ref{fig:activation_result}. From Figure~\ref{fig:activation_result}, firstly we can see that ReLU and the three further functions related to ReLU have faster convergence speed than Sigmoid function both on the training dataset and validation dataset. I guess the reason of the poor performance of Sigmoid function may be that it often causes the derivative to gradually change to 0 in the neural network. So the parameters can not be updated and the neural network can not be optimized. I think it is a good choice to use ReLU and its further functions as the activation function. Secondly, comparing the results of the ReLU family (ReLU, Leaky ReLU, ELU and SELU), the differences between these functions is very small, but we can see that the convergence of the three further functions related to ReLU is slightly faster than the ReLU function. Meanwhile, from Table~\ref{tab:activationResult}, we can see the errors of Leaky ReLU, ELU and SELU are a little lower than ReLU. Maybe I can assume that the Leaky ReLU, ELU and SELU functions are better choices when selecting the activation function.
