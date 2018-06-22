
# coding: utf-8

# # Coursework 1
# 
# This notebook is intended to be used as a starting point for your experiments. The instructions can be found in the instructions file located under spec/coursework1.pdf. The methods provided here are just helper functions. If you want more complex graphs such as side by side comparisons of different experiments you should learn more about matplotlib and implement them. Before each experiment remember to re-initialize neural network weights and reset the data providers so you get a properly initialized experiment. For each experiment try to keep most hyperparameters the same except the one under investigation so you can understand what the effects of each are.

# In[1]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True):
    
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    
    return stats, keys


# In[2]:


# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider

import os
os.environ["MLP_DATA_DIR"] = "..\data"

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)
batch_size = 100
# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=batch_size, rng=rng)


# ## Part2A  
# ### different activation layers such as  SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer....

# In[3]:


from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit, SELUInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 10, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)


model_SigmoidLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_ReluLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_LeakyReluLayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_ELULayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    ELULayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    ELULayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_SELULayer = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    SELULayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    SELULayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])




error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('SigmoidLayer\n\n')
[stats_SigmoidLayer, keys_SigmoidLayer] = train_model_and_plot_stats(
    model_SigmoidLayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('ReluLayer\n\n')
[stats_ReluLayer, keys_ReluLayer] = train_model_and_plot_stats(
    model_ReluLayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('LeakyReluLayer\n\n')
[stats_LeakyReluLayer, keys_LeakyReluLayer] = train_model_and_plot_stats(
    model_LeakyReluLayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('ELULayer\n\n')
[stats_ELULayer, keys_ELULayer] = train_model_and_plot_stats(
    model_ELULayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('SELULayer\n\n')
[stats_SELULayer, keys_SELULayer] = train_model_and_plot_stats(
    model_SELULayer, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)


 # Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats_SigmoidLayer.shape[0]) * stats_interval,
            stats_SigmoidLayer[1:, keys_SigmoidLayer[k]], label='SigmoidLayer')
    ax_1.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
            stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')

    ax_1.plot(np.arange(1, stats_LeakyReluLayer.shape[0]) * stats_interval,
            stats_LeakyReluLayer[1:, keys_LeakyReluLayer[k]], label='LeakyReluLayer')

    ax_1.plot(np.arange(1, stats_ELULayer.shape[0]) * stats_interval,
            stats_ELULayer[1:, keys_ELULayer[k]], label='ELULayer')

    ax_1.plot(np.arange(1, stats_SELULayer.shape[0]) * stats_interval,
            stats_SELULayer[1:, keys_SELULayer[k]], label='SELULayer')
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of different activation layers  in the training dataset')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('Part2A_activation_train_error.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_2.plot(np.arange(1, stats_SigmoidLayer.shape[0]) * stats_interval,
            stats_SigmoidLayer[1:, keys_SigmoidLayer[k]], label='SigmoidLayer')
    ax_2.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
            stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')

    ax_2.plot(np.arange(1, stats_LeakyReluLayer.shape[0]) * stats_interval,
            stats_LeakyReluLayer[1:, keys_LeakyReluLayer[k]], label='LeakyReluLayer')

    ax_2.plot(np.arange(1, stats_ELULayer.shape[0]) * stats_interval,
            stats_ELULayer[1:, keys_ELULayer[k]], label='ELULayer')

    ax_2.plot(np.arange(1, stats_SELULayer.shape[0]) * stats_interval,
            stats_SELULayer[1:, keys_SELULayer[k]], label='SELULayer')
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of different activation layers  in the valid dataset')
fig_2.tight_layout() # This minimises whitespace around the axes.
fig_2.savefig('Part2A_activation_valid_error.pdf') # Save figure to current directory in PDF format
#


fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_3.plot(np.arange(1, stats_SigmoidLayer.shape[0]) * stats_interval,
            stats_SigmoidLayer[1:, keys_SigmoidLayer[k]], label='SigmoidLayer')
    ax_3.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
            stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')

    ax_3.plot(np.arange(1, stats_LeakyReluLayer.shape[0]) * stats_interval,
            stats_LeakyReluLayer[1:, keys_LeakyReluLayer[k]], label='LeakyReluLayer')

    ax_3.plot(np.arange(1, stats_ELULayer.shape[0]) * stats_interval,
            stats_ELULayer[1:, keys_ELULayer[k]], label='ELULayer')

    ax_3.plot(np.arange(1, stats_SELULayer.shape[0]) * stats_interval,
            stats_SELULayer[1:, keys_SELULayer[k]], label='SELULayer')
ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of different activation layers  in the training dataset')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('Part2A_activation_train_acc.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, stats_SigmoidLayer.shape[0]) * stats_interval,
            stats_SigmoidLayer[1:, keys_SigmoidLayer[k]], label='SigmoidLayer')
    ax_4.plot(np.arange(1, stats_ReluLayer.shape[0]) * stats_interval,
            stats_ReluLayer[1:, keys_ReluLayer[k]], label='ReluLayer')

    ax_4.plot(np.arange(1, stats_LeakyReluLayer.shape[0]) * stats_interval,
            stats_LeakyReluLayer[1:, keys_LeakyReluLayer[k]], label='LeakyReluLayer')

    ax_4.plot(np.arange(1, stats_ELULayer.shape[0]) * stats_interval,
            stats_ELULayer[1:, keys_ELULayer[k]], label='ELULayer')

    ax_4.plot(np.arange(1, stats_SELULayer.shape[0]) * stats_interval,
            stats_SELULayer[1:, keys_SELULayer[k]], label='SELULayer')
ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of different activation layers  in the valid dataset')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('Part2A_activation_valid_acc.pdf') # Save figure to current directory in PDF format
#


# ## Part2B 
# ### Leaky ReLU compare networks with 2–8 hidden layers, using 100 hidden units per hidden layer.

# In[4]:


# ### Part2B
# Leaky ReLU compare networks with 2–8 hidden layers, using 100 hidden units per hidden layer.

from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit, SELUInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 10, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)



model_LeakyReluLayer2 = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])


model_LeakyReluLayer3 = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_LeakyReluLayer4 = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_LeakyReluLayer5 = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_LeakyReluLayer6 = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_LeakyReluLayer7 = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

model_LeakyReluLayer8 = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])






error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('LeakyReluLayer 2\n\n')
[stats_LeakyReluLayer2, keys_LeakyReluLayer2] = train_model_and_plot_stats(
    model_LeakyReluLayer2, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)


print('LeakyReluLayer 3\n\n')
[stats_LeakyReluLayer3, keys_LeakyReluLayer3] = train_model_and_plot_stats(
    model_LeakyReluLayer3, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('LeakyReluLayer 4\n\n')
[stats_LeakyReluLayer4, keys_LeakyReluLayer4] = train_model_and_plot_stats(
    model_LeakyReluLayer4, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('LeakyReluLayer 5\n\n')
[stats_LeakyReluLayer5, keys_LeakyReluLayer5] = train_model_and_plot_stats(
    model_LeakyReluLayer5, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('LeakyReluLayer 6\n\n')
[stats_LeakyReluLayer6, keys_LeakyReluLayer6] = train_model_and_plot_stats(
    model_LeakyReluLayer6, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('LeakyReluLayer 7\n\n')
[stats_LeakyReluLayer7, keys_LeakyReluLayer7] = train_model_and_plot_stats(
    model_LeakyReluLayer7, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

print('LeakyReluLayer 8\n\n')
[stats_LeakyReluLayer8, keys_LeakyReluLayer8] = train_model_and_plot_stats(
    model_LeakyReluLayer8, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)



 # Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats_LeakyReluLayer2.shape[0]) * stats_interval,
            stats_LeakyReluLayer2[1:, keys_LeakyReluLayer2[k]], label='2 layers')

    ax_1.plot(np.arange(1, stats_LeakyReluLayer3.shape[0]) * stats_interval,
            stats_LeakyReluLayer3[1:, keys_LeakyReluLayer3[k]], label='3 layers')

    ax_1.plot(np.arange(1, stats_LeakyReluLayer4.shape[0]) * stats_interval,
            stats_LeakyReluLayer4[1:, keys_LeakyReluLayer4[k]], label='4 layers')

    ax_1.plot(np.arange(1, stats_LeakyReluLayer5.shape[0]) * stats_interval,
            stats_LeakyReluLayer5[1:, keys_LeakyReluLayer5[k]], label='5 layers')

    ax_1.plot(np.arange(1, stats_LeakyReluLayer6.shape[0]) * stats_interval,
            stats_LeakyReluLayer6[1:, keys_LeakyReluLayer6[k]], label='6 layers')

    ax_1.plot(np.arange(1, stats_LeakyReluLayer7.shape[0]) * stats_interval,
            stats_LeakyReluLayer7[1:, keys_LeakyReluLayer7[k]], label='7 layers')

    ax_1.plot(np.arange(1, stats_LeakyReluLayer8.shape[0]) * stats_interval,
            stats_LeakyReluLayer8[1:, keys_LeakyReluLayer8[k]], label='8 layers')

ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of different layers in the training dataset')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('Part2B_layers_train_error.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_2.plot(np.arange(1, stats_LeakyReluLayer2.shape[0]) * stats_interval,
            stats_LeakyReluLayer2[1:, keys_LeakyReluLayer2[k]], label='2 layers')

    ax_2.plot(np.arange(1, stats_LeakyReluLayer3.shape[0]) * stats_interval,
            stats_LeakyReluLayer3[1:, keys_LeakyReluLayer3[k]], label='3 layers')

    ax_2.plot(np.arange(1, stats_LeakyReluLayer4.shape[0]) * stats_interval,
            stats_LeakyReluLayer4[1:, keys_LeakyReluLayer4[k]], label='4 layers')

    ax_2.plot(np.arange(1, stats_LeakyReluLayer5.shape[0]) * stats_interval,
            stats_LeakyReluLayer5[1:, keys_LeakyReluLayer5[k]], label='5 layers')

    ax_2.plot(np.arange(1, stats_LeakyReluLayer6.shape[0]) * stats_interval,
            stats_LeakyReluLayer6[1:, keys_LeakyReluLayer6[k]], label='6 layers')

    ax_2.plot(np.arange(1, stats_LeakyReluLayer7.shape[0]) * stats_interval,
            stats_LeakyReluLayer7[1:, keys_LeakyReluLayer7[k]], label='7 layers')

    ax_2.plot(np.arange(1, stats_LeakyReluLayer8.shape[0]) * stats_interval,
            stats_LeakyReluLayer8[1:, keys_LeakyReluLayer8[k]], label='8 layers')

ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of different layers  in the valid dataset')
fig_2.tight_layout() # This minimises whitespace around the axes.
fig_2.savefig('Part2B_layers_valid_error.pdf') # Save figure to current directory in PDF format
#


fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_3.plot(np.arange(1, stats_LeakyReluLayer2.shape[0]) * stats_interval,
            stats_LeakyReluLayer2[1:, keys_LeakyReluLayer2[k]], label='2 layers')

    ax_3.plot(np.arange(1, stats_LeakyReluLayer3.shape[0]) * stats_interval,
            stats_LeakyReluLayer3[1:, keys_LeakyReluLayer3[k]], label='3 layers')

    ax_3.plot(np.arange(1, stats_LeakyReluLayer4.shape[0]) * stats_interval,
            stats_LeakyReluLayer4[1:, keys_LeakyReluLayer4[k]], label='4 layers')

    ax_3.plot(np.arange(1, stats_LeakyReluLayer5.shape[0]) * stats_interval,
            stats_LeakyReluLayer5[1:, keys_LeakyReluLayer5[k]], label='5 layers')

    ax_3.plot(np.arange(1, stats_LeakyReluLayer6.shape[0]) * stats_interval,
            stats_LeakyReluLayer6[1:, keys_LeakyReluLayer6[k]], label='6 layers')

    ax_3.plot(np.arange(1, stats_LeakyReluLayer7.shape[0]) * stats_interval,
            stats_LeakyReluLayer7[1:, keys_LeakyReluLayer7[k]], label='7 layers')

    ax_3.plot(np.arange(1, stats_LeakyReluLayer8.shape[0]) * stats_interval,
            stats_LeakyReluLayer8[1:, keys_LeakyReluLayer8[k]], label='8 layers')

ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of different layers  in the training dataset')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('Part2B_layers_train_acc.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, stats_LeakyReluLayer2.shape[0]) * stats_interval,
            stats_LeakyReluLayer2[1:, keys_LeakyReluLayer2[k]], label='2 layers')

    ax_4.plot(np.arange(1, stats_LeakyReluLayer3.shape[0]) * stats_interval,
            stats_LeakyReluLayer3[1:, keys_LeakyReluLayer3[k]], label='3 layers')

    ax_4.plot(np.arange(1, stats_LeakyReluLayer4.shape[0]) * stats_interval,
            stats_LeakyReluLayer4[1:, keys_LeakyReluLayer4[k]], label='4 layers')

    ax_4.plot(np.arange(1, stats_LeakyReluLayer5.shape[0]) * stats_interval,
            stats_LeakyReluLayer5[1:, keys_LeakyReluLayer5[k]], label='5 layers')

    ax_4.plot(np.arange(1, stats_LeakyReluLayer6.shape[0]) * stats_interval,
            stats_LeakyReluLayer6[1:, keys_LeakyReluLayer6[k]], label='6 layers')

    ax_4.plot(np.arange(1, stats_LeakyReluLayer7.shape[0]) * stats_interval,
            stats_LeakyReluLayer7[1:, keys_LeakyReluLayer7[k]], label='7 layers')

    ax_4.plot(np.arange(1, stats_LeakyReluLayer8.shape[0]) * stats_interval,
            stats_LeakyReluLayer8[1:, keys_LeakyReluLayer8[k]], label='8 layers')

ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of different layers in the valid dataset')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('Part2B_layers_valid_acc.pdf') # Save figure to current directory in PDF format
#


# ## Part2B
# ### Part2B, we will use different weight initialisations to train network.
# 

# In[5]:



from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit, SELUInit,GlorotUniformInitFanIn,GlorotUniformInitFanOut,GlorotNormalInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser

#setup hyperparameters
learning_rate = 0.1
num_epochs = 100
stats_interval = 1
input_dim, output_dim, hidden_dim = 784, 10, 100

weights_initFan_inFan_out = GlorotUniformInit(rng=rng)
weights_initFan_in = GlorotUniformInitFanIn(rng=rng)
weights_initFan_out = GlorotUniformInitFanOut(rng=rng)
weights_initNormal = GlorotNormalInit(rng=rng)
weights_initSELU = SELUInit(rng=rng)

biases_init = ConstantInit(0.)
model_LeakyReluLayerFan_inFan_out = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_initFan_inFan_out, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_initFan_inFan_out, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_initFan_inFan_out, biases_init)
])


model_LeakyReluLayerFan_in = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_initFan_in, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_initFan_in, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_initFan_in, biases_init)
])

model_LeakyReluLayerFan_out = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_initFan_out, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_initFan_out, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_initFan_out, biases_init)
])

model_LeakyReluLayerNormal = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_initNormal, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_initNormal, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_initNormal, biases_init)
])

model_LeakyReluLayerSELU = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_initSELU, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_initSELU, biases_init),
    LeakyReluLayer(),
    AffineLayer(hidden_dim, output_dim, weights_initSELU, biases_init)
])


error = CrossEntropySoftmaxError()
# Use a basic gradient descent learning rule
learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

#Remember to use notebook=False when you write a script to be run in a terminal
print('LeakyReluLayer 2\n\n')
[stats_LeakyReluLayerFan_inFan_out, keys_LeakyReluLayerFan_inFan_out] = train_model_and_plot_stats(
    model_LeakyReluLayerFan_inFan_out, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

[stats_LeakyReluLayerFan_in, keys_LeakyReluLayerFan_in] = train_model_and_plot_stats(
    model_LeakyReluLayerFan_in, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

[stats_LeakyReluLayerFan_out, keys_LeakyReluLayerFan_out] = train_model_and_plot_stats(
    model_LeakyReluLayerFan_out, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

[stats_LeakyReluLayerNormal, keys_LeakyReluLayerNormal] = train_model_and_plot_stats(
    model_LeakyReluLayerNormal, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

[stats_LeakyReluLayerSELU, keys_LeakyReluLayerSELU] = train_model_and_plot_stats(
    model_LeakyReluLayerSELU, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)




 # Plot the change in the validation and training set error over training.
# Plot Errors of different activation layers  in the training dataset
fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)']:
    ax_1.plot(np.arange(1, stats_LeakyReluLayerFan_inFan_out.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_inFan_out[1:, keys_LeakyReluLayerFan_inFan_out[k]], label='Fan_inFan_out')

    ax_1.plot(np.arange(1, stats_LeakyReluLayerFan_in.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_in[1:, keys_LeakyReluLayerFan_in[k]], label='Fan_in')

    ax_1.plot(np.arange(1, stats_LeakyReluLayerFan_out.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_out[1:, keys_LeakyReluLayerFan_out[k]], label='Fan_out')

    ax_1.plot(np.arange(1, stats_LeakyReluLayerNormal.shape[0]) * stats_interval,
            stats_LeakyReluLayerNormal[1:, keys_LeakyReluLayerNormal[k]], label='Normal')

    ax_1.plot(np.arange(1, stats_LeakyReluLayerSELU.shape[0]) * stats_interval,
            stats_LeakyReluLayerSELU[1:, keys_LeakyReluLayerSELU[k]], label='SELU')



ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')
ax_1.set_ylabel('Error(train)')

ax_1.set_title('Errors of different initialisations in the training dataset')
fig_1.tight_layout() # This minimises whitespace around the axes.
fig_1.savefig('Part2B_initial_train_error.pdf') # Save figure to current directory in PDF format
#

fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['error(valid)']:
    ax_2.plot(np.arange(1, stats_LeakyReluLayerFan_inFan_out.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_inFan_out[1:, keys_LeakyReluLayerFan_inFan_out[k]], label='Fan_inFan_out')

    ax_2.plot(np.arange(1, stats_LeakyReluLayerFan_in.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_in[1:, keys_LeakyReluLayerFan_in[k]], label='Fan_in')

    ax_2.plot(np.arange(1, stats_LeakyReluLayerFan_out.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_out[1:, keys_LeakyReluLayerFan_out[k]], label='Fan_out')

    ax_2.plot(np.arange(1, stats_LeakyReluLayerNormal.shape[0]) * stats_interval,
            stats_LeakyReluLayerNormal[1:, keys_LeakyReluLayerNormal[k]], label='Normal')

    ax_2.plot(np.arange(1, stats_LeakyReluLayerSELU.shape[0]) * stats_interval,
            stats_LeakyReluLayerSELU[1:, keys_LeakyReluLayerSELU[k]], label='SELU')

ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Error(valid)')

ax_2.set_title('Errors of different initialisations  in the valid dataset')
fig_2.tight_layout() # This minimises whitespace around the axes.
fig_2.savefig('Part2B_initial_valid_error.pdf') # Save figure to current directory in PDF format
#


fig_3 = plt.figure(figsize=(8, 4))
ax_3 = fig_3.add_subplot(111)
for k in ['acc(train)']:
    ax_3.plot(np.arange(1, stats_LeakyReluLayerFan_inFan_out.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_inFan_out[1:, keys_LeakyReluLayerFan_inFan_out[k]], label='Fan_inFan_out')

    ax_3.plot(np.arange(1, stats_LeakyReluLayerFan_in.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_in[1:, keys_LeakyReluLayerFan_in[k]], label='Fan_in')

    ax_3.plot(np.arange(1, stats_LeakyReluLayerFan_out.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_out[1:, keys_LeakyReluLayerFan_out[k]], label='Fan_out')

    ax_3.plot(np.arange(1, stats_LeakyReluLayerNormal.shape[0]) * stats_interval,
            stats_LeakyReluLayerNormal[1:, keys_LeakyReluLayerNormal[k]], label='Normal')

    ax_3.plot(np.arange(1, stats_LeakyReluLayerSELU.shape[0]) * stats_interval,
            stats_LeakyReluLayerSELU[1:, keys_LeakyReluLayerSELU[k]], label='SELU')

ax_3.legend(loc=0)
ax_3.set_xlabel('Epoch number')
ax_3.set_ylabel('Accuracy(train)')

ax_3.set_title('Accuracies of different initialisations  in the training dataset')
fig_3.tight_layout() # This minimises whitespace around the axes.
fig_3.savefig('Part2B_initial_train_acc.pdf') # Save figure to current directory in PDF format
#

fig_4 = plt.figure(figsize=(8, 4))
ax_4 = fig_4.add_subplot(111)
for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, stats_LeakyReluLayerFan_inFan_out.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_inFan_out[1:, keys_LeakyReluLayerFan_inFan_out[k]], label='Fan_inFan_out')

    ax_4.plot(np.arange(1, stats_LeakyReluLayerFan_in.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_in[1:, keys_LeakyReluLayerFan_in[k]], label='Fan_in')

    ax_4.plot(np.arange(1, stats_LeakyReluLayerFan_out.shape[0]) * stats_interval,
            stats_LeakyReluLayerFan_out[1:, keys_LeakyReluLayerFan_out[k]], label='Fan_out')

    ax_4.plot(np.arange(1, stats_LeakyReluLayerNormal.shape[0]) * stats_interval,
            stats_LeakyReluLayerNormal[1:, keys_LeakyReluLayerNormal[k]], label='Normal')

    ax_4.plot(np.arange(1, stats_LeakyReluLayerSELU.shape[0]) * stats_interval,
            stats_LeakyReluLayerSELU[1:, keys_LeakyReluLayerSELU[k]], label='SELU')

ax_4.legend(loc=0)
ax_4.set_xlabel('Epoch number')
ax_4.set_ylabel('Accuracy(valid)')

ax_4.set_title('Accuracies of different initialisations in the valid dataset')
fig_4.tight_layout() # This minimises whitespace around the axes.
fig_4.savefig('Part2B_initial_valid_acc.pdf') # Save figure to current directory in PDF format
#

