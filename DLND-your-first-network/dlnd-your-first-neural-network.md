
# Your first neural network

In this project, you'll build your first neural network and use it to predict daily bike rental ridership. We've provided some of the code, but left the implementation of the neural network up to you (for the most part). After you've submitted this project, feel free to explore the data and the model more.




```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Load and prepare the data

A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, we've written the code to load and prepare the data. You'll learn more about this soon!


```python
data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)
```


```python
rides.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instant</th>
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Checking out the data

This dataset has the number of riders for each hour of each day from January 1 2011 to December 31 2012. The number of riders is split between casual and registered, summed up in the `cnt` column. You can see the first few rows of the data above.

Below is a plot showing the number of bike riders over the first 10 days in the data set. You can see the hourly rentals here. This data is pretty complicated! The weekends have lower over all ridership and there are spikes when people are biking to and from work during the week. Looking at the data above, we also have information about temperature, humidity, and windspeed, all of these likely affecting the number of riders. You'll be trying to capture all this with your model.


```python
rides[:24*10].plot(x='dteday', y='cnt')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xc181780>




![png](output_6_1.png)


### Dummy variables
Here we have some categorical variables like season, weather, month. To include these in our model, we'll need to make binary dummy variables. This is simple to do with Pandas thanks to `get_dummies()`.


```python
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yr</th>
      <th>holiday</th>
      <th>temp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
      <th>season_1</th>
      <th>season_2</th>
      <th>...</th>
      <th>hr_21</th>
      <th>hr_22</th>
      <th>hr_23</th>
      <th>weekday_0</th>
      <th>weekday_1</th>
      <th>weekday_2</th>
      <th>weekday_3</th>
      <th>weekday_4</th>
      <th>weekday_5</th>
      <th>weekday_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.24</td>
      <td>0.81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0.22</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0.22</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0.24</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0.24</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>



### Scaling target variables
To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.

The scaling factors are saved so we can go backwards when we use the network for predictions.


```python
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
```

### Splitting the data into training, testing, and validation sets

We'll save the last 21 days of the data to use as a test set after we've trained the network. We'll use this set to make predictions and compare them with the actual number of riders.


```python
# Save the last 21 days 
test_data = data[-21*24:]
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
```

We'll split the data into two sets, one for training and one for validating as the network is being trained. Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set).


```python
# Hold out the last 60 days of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]
```

## Time to build the network

Below you'll build your network. We've built out the structure and the backwards pass. You'll implement the forward pass through the network. You'll also set the hyperparameters: the learning rate, the number of hidden units, and the number of training passes.

The network has two layers, a hidden layer and an output layer. The hidden layer will use the sigmoid function for activations. The output layer has only one node and is used for the regression, the output of the node is the same as the input of the node. That is, the activation function is $f(x)=x$. A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an activation function. We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called *forward propagation*.

We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called *backpropagation*.

> **Hint:** You'll need the derivative of the output activation function ($f(x) = x$) for the backpropagation implementation. If you aren't familiar with calculus, this function is equivalent to the equation $y = x$. What is the slope of that equation? That is the derivative of $f(x)$.

Below, you have these tasks:
1. Implement the sigmoid function to use as the activation function. Set `self.activation_function` in `__init__` to your sigmoid function.
2. Implement the forward pass in the `train` method.
3. Implement the backpropagation algorithm in the `train` method, including calculating the output error.
4. Implement the forward pass in the `run` method.
  


```python
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)# signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # TODO: Output error
        output_errors = targets - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # TODO: Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors) * hidden_outputs * (1 - hidden_outputs) # errors propagated to the hidden layer
        hidden_grad = hidden_errors *  inputs.T # hidden layer gradients
        
        # TODO: Update the weights
        self.weights_hidden_to_output += self.lr * output_errors * hidden_outputs.T# update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * hidden_grad # update input-to-hidden weights with gradient descent step
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs
```


```python
def MSE(y, Y):
    return np.mean((y-Y)**2)
```

## Training the network

Here you'll set the hyperparameters for the network. The strategy here is to find hyperparameters such that the error on the training set is low, but you're not overfitting to the data. If you train the network too long or have too many hidden nodes, it can become overly specific to the training set and will fail to generalize to the validation set. That is, the loss on the validation set will start increasing as the training set loss drops.

You'll also be using a method know as Stochastic Gradient Descent (SGD) to train the network. The idea is that for each training pass, you grab a random sample of the data instead of using the whole data set. You use many more training passes than with normal gradient descent, but each pass is much faster. This ends up training the network more efficiently. You'll learn more about SGD later.

### Choose the number of epochs
This is the number of times the dataset will pass through the network, each time updating the weights. As the number of epochs increases, the network becomes better and better at predicting the targets in the training set. You'll need to choose enough epochs to train the network well but not too many or you'll be overfitting.

### Choose the learning rate
This scales the size of weight updates. If this is too big, the weights tend to explode and the network fails to fit the data. A good choice to start at is 0.1. If the network has problems fitting the data, try reducing the learning rate. Note that the lower the learning rate, the smaller the steps are in the weight updates and the longer it takes for the neural network to converge.

### Choose the number of hidden nodes
The more hidden nodes you have, the more accurate predictions the model will make. Try a few different numbers and see how it affects the performance. You can look at the losses dictionary for a metric of the network performance. If the number of hidden units is too low, then the model won't have enough space to learn and if it is too high there are too many options for the direction that the learning can take. The trick here is to find the right balance in number of hidden units you choose.


```python
import sys

### Set the hyperparameters here ###
epochs = 200
learning_rate = 0.1
hidden_nodes = 7
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values, 
                              train_targets.ix[batch]['cnt']):
        network.train(record, target)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\r\nProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
```

    
    Progress: 0.0% ... Training loss: 0.923 ... Validation loss: 2.073
    Progress: 0.5% ... Training loss: 0.725 ... Validation loss: 1.339
    Progress: 1.0% ... Training loss: 0.603 ... Validation loss: 1.024
    Progress: 1.5% ... Training loss: 0.580 ... Validation loss: 0.720
    Progress: 2.0% ... Training loss: 0.466 ... Validation loss: 0.770
    Progress: 2.5% ... Training loss: 0.475 ... Validation loss: 0.714
    Progress: 3.0% ... Training loss: 0.419 ... Validation loss: 0.699
    Progress: 3.5% ... Training loss: 0.369 ... Validation loss: 0.584
    Progress: 4.0% ... Training loss: 0.418 ... Validation loss: 0.627
    Progress: 4.5% ... Training loss: 0.410 ... Validation loss: 0.747
    Progress: 5.0% ... Training loss: 0.352 ... Validation loss: 0.562
    Progress: 5.5% ... Training loss: 0.383 ... Validation loss: 0.581
    Progress: 6.0% ... Training loss: 0.362 ... Validation loss: 0.544
    Progress: 6.5% ... Training loss: 0.343 ... Validation loss: 0.496
    Progress: 7.0% ... Training loss: 0.550 ... Validation loss: 0.588
    Progress: 7.5% ... Training loss: 0.389 ... Validation loss: 0.524
    Progress: 8.0% ... Training loss: 0.306 ... Validation loss: 0.466
    Progress: 8.5% ... Training loss: 0.313 ... Validation loss: 0.509
    Progress: 9.0% ... Training loss: 0.466 ... Validation loss: 0.618
    Progress: 9.5% ... Training loss: 0.344 ... Validation loss: 0.658
    Progress: 10.0% ... Training loss: 0.368 ... Validation loss: 0.608
    Progress: 10.5% ... Training loss: 0.312 ... Validation loss: 0.499
    Progress: 11.0% ... Training loss: 0.326 ... Validation loss: 0.507
    Progress: 11.5% ... Training loss: 0.351 ... Validation loss: 0.493
    Progress: 12.0% ... Training loss: 0.346 ... Validation loss: 0.486
    Progress: 12.5% ... Training loss: 0.305 ... Validation loss: 0.483
    Progress: 13.0% ... Training loss: 0.363 ... Validation loss: 0.494
    Progress: 13.5% ... Training loss: 0.351 ... Validation loss: 0.523
    Progress: 14.0% ... Training loss: 0.377 ... Validation loss: 0.500
    Progress: 14.5% ... Training loss: 0.343 ... Validation loss: 0.463
    Progress: 15.0% ... Training loss: 0.291 ... Validation loss: 0.502
    Progress: 15.5% ... Training loss: 0.301 ... Validation loss: 0.477
    Progress: 16.0% ... Training loss: 0.312 ... Validation loss: 0.534
    Progress: 16.5% ... Training loss: 0.293 ... Validation loss: 0.482
    Progress: 17.0% ... Training loss: 0.289 ... Validation loss: 0.463
    Progress: 17.5% ... Training loss: 0.270 ... Validation loss: 0.444
    Progress: 18.0% ... Training loss: 0.282 ... Validation loss: 0.476
    Progress: 18.5% ... Training loss: 0.272 ... Validation loss: 0.449
    Progress: 19.0% ... Training loss: 0.253 ... Validation loss: 0.429
    Progress: 19.5% ... Training loss: 0.286 ... Validation loss: 0.452
    Progress: 20.0% ... Training loss: 0.251 ... Validation loss: 0.525
    Progress: 20.5% ... Training loss: 0.250 ... Validation loss: 0.423
    Progress: 21.0% ... Training loss: 0.255 ... Validation loss: 0.416
    Progress: 21.5% ... Training loss: 0.273 ... Validation loss: 0.396
    Progress: 22.0% ... Training loss: 0.247 ... Validation loss: 0.404
    Progress: 22.5% ... Training loss: 0.293 ... Validation loss: 0.438
    Progress: 23.0% ... Training loss: 0.260 ... Validation loss: 0.476
    Progress: 23.5% ... Training loss: 0.230 ... Validation loss: 0.411
    Progress: 24.0% ... Training loss: 0.229 ... Validation loss: 0.350
    Progress: 24.5% ... Training loss: 0.237 ... Validation loss: 0.404
    Progress: 25.0% ... Training loss: 0.243 ... Validation loss: 0.501
    Progress: 25.5% ... Training loss: 0.217 ... Validation loss: 0.373
    Progress: 26.0% ... Training loss: 0.230 ... Validation loss: 0.384
    Progress: 26.5% ... Training loss: 0.233 ... Validation loss: 0.386
    Progress: 27.0% ... Training loss: 0.230 ... Validation loss: 0.407
    Progress: 27.5% ... Training loss: 0.230 ... Validation loss: 0.346
    Progress: 28.0% ... Training loss: 0.209 ... Validation loss: 0.361
    Progress: 28.5% ... Training loss: 0.208 ... Validation loss: 0.361
    Progress: 29.0% ... Training loss: 0.210 ... Validation loss: 0.367
    Progress: 29.5% ... Training loss: 0.201 ... Validation loss: 0.335
    Progress: 30.0% ... Training loss: 0.220 ... Validation loss: 0.366
    Progress: 30.5% ... Training loss: 0.220 ... Validation loss: 0.394
    Progress: 31.0% ... Training loss: 0.206 ... Validation loss: 0.354
    Progress: 31.5% ... Training loss: 0.217 ... Validation loss: 0.320
    Progress: 32.0% ... Training loss: 0.256 ... Validation loss: 0.395
    Progress: 32.5% ... Training loss: 0.190 ... Validation loss: 0.340
    Progress: 33.0% ... Training loss: 0.190 ... Validation loss: 0.337
    Progress: 33.5% ... Training loss: 0.195 ... Validation loss: 0.348
    Progress: 34.0% ... Training loss: 0.188 ... Validation loss: 0.343
    Progress: 34.5% ... Training loss: 0.193 ... Validation loss: 0.329
    Progress: 35.0% ... Training loss: 0.235 ... Validation loss: 0.323
    Progress: 35.5% ... Training loss: 0.181 ... Validation loss: 0.314
    Progress: 36.0% ... Training loss: 0.178 ... Validation loss: 0.342
    Progress: 36.5% ... Training loss: 0.199 ... Validation loss: 0.357
    Progress: 37.0% ... Training loss: 0.180 ... Validation loss: 0.312
    Progress: 37.5% ... Training loss: 0.229 ... Validation loss: 0.325
    Progress: 38.0% ... Training loss: 0.174 ... Validation loss: 0.412
    Progress: 38.5% ... Training loss: 0.164 ... Validation loss: 0.315
    Progress: 39.0% ... Training loss: 0.159 ... Validation loss: 0.328
    Progress: 39.5% ... Training loss: 0.195 ... Validation loss: 0.378
    Progress: 40.0% ... Training loss: 0.155 ... Validation loss: 0.332
    Progress: 40.5% ... Training loss: 0.219 ... Validation loss: 0.381
    Progress: 41.0% ... Training loss: 0.165 ... Validation loss: 0.401
    Progress: 41.5% ... Training loss: 0.176 ... Validation loss: 0.287
    Progress: 42.0% ... Training loss: 0.172 ... Validation loss: 0.278
    Progress: 42.5% ... Training loss: 0.164 ... Validation loss: 0.326
    Progress: 43.0% ... Training loss: 0.149 ... Validation loss: 0.314
    Progress: 43.5% ... Training loss: 0.153 ... Validation loss: 0.351
    Progress: 44.0% ... Training loss: 0.149 ... Validation loss: 0.383
    Progress: 44.5% ... Training loss: 0.147 ... Validation loss: 0.308
    Progress: 45.0% ... Training loss: 0.189 ... Validation loss: 0.304
    Progress: 45.5% ... Training loss: 0.152 ... Validation loss: 0.274
    Progress: 46.0% ... Training loss: 0.183 ... Validation loss: 0.291
    Progress: 46.5% ... Training loss: 0.149 ... Validation loss: 0.278
    Progress: 47.0% ... Training loss: 0.144 ... Validation loss: 0.240
    Progress: 47.5% ... Training loss: 0.186 ... Validation loss: 0.265
    Progress: 48.0% ... Training loss: 0.135 ... Validation loss: 0.255
    Progress: 48.5% ... Training loss: 0.135 ... Validation loss: 0.257
    Progress: 49.0% ... Training loss: 0.130 ... Validation loss: 0.234
    Progress: 49.5% ... Training loss: 0.123 ... Validation loss: 0.215
    Progress: 50.0% ... Training loss: 0.118 ... Validation loss: 0.218
    Progress: 50.5% ... Training loss: 0.193 ... Validation loss: 0.268
    Progress: 51.0% ... Training loss: 0.133 ... Validation loss: 0.229
    Progress: 51.5% ... Training loss: 0.148 ... Validation loss: 0.251
    Progress: 52.0% ... Training loss: 0.130 ... Validation loss: 0.265
    Progress: 52.5% ... Training loss: 0.137 ... Validation loss: 0.228
    Progress: 53.0% ... Training loss: 0.134 ... Validation loss: 0.240
    Progress: 53.5% ... Training loss: 0.144 ... Validation loss: 0.217
    Progress: 54.0% ... Training loss: 0.110 ... Validation loss: 0.220
    Progress: 54.5% ... Training loss: 0.128 ... Validation loss: 0.207
    Progress: 55.0% ... Training loss: 0.126 ... Validation loss: 0.202
    Progress: 55.5% ... Training loss: 0.112 ... Validation loss: 0.230
    Progress: 56.0% ... Training loss: 0.113 ... Validation loss: 0.270
    Progress: 56.5% ... Training loss: 0.114 ... Validation loss: 0.197
    Progress: 57.0% ... Training loss: 0.121 ... Validation loss: 0.197
    Progress: 57.5% ... Training loss: 0.140 ... Validation loss: 0.229
    Progress: 58.0% ... Training loss: 0.127 ... Validation loss: 0.291
    Progress: 58.5% ... Training loss: 0.112 ... Validation loss: 0.190
    Progress: 59.0% ... Training loss: 0.106 ... Validation loss: 0.182
    Progress: 59.5% ... Training loss: 0.123 ... Validation loss: 0.299
    Progress: 60.0% ... Training loss: 0.111 ... Validation loss: 0.221
    Progress: 60.5% ... Training loss: 0.157 ... Validation loss: 0.208
    Progress: 61.0% ... Training loss: 0.125 ... Validation loss: 0.210
    Progress: 61.5% ... Training loss: 0.100 ... Validation loss: 0.200
    Progress: 62.0% ... Training loss: 0.119 ... Validation loss: 0.219
    Progress: 62.5% ... Training loss: 0.102 ... Validation loss: 0.223
    Progress: 63.0% ... Training loss: 0.112 ... Validation loss: 0.209
    Progress: 63.5% ... Training loss: 0.109 ... Validation loss: 0.213
    Progress: 64.0% ... Training loss: 0.121 ... Validation loss: 0.232
    Progress: 64.5% ... Training loss: 0.105 ... Validation loss: 0.173
    Progress: 65.0% ... Training loss: 0.106 ... Validation loss: 0.179
    Progress: 65.5% ... Training loss: 0.091 ... Validation loss: 0.170
    Progress: 66.0% ... Training loss: 0.101 ... Validation loss: 0.168
    Progress: 66.5% ... Training loss: 0.099 ... Validation loss: 0.186
    Progress: 67.0% ... Training loss: 0.106 ... Validation loss: 0.183
    Progress: 67.5% ... Training loss: 0.111 ... Validation loss: 0.202
    Progress: 68.0% ... Training loss: 0.110 ... Validation loss: 0.257
    Progress: 68.5% ... Training loss: 0.087 ... Validation loss: 0.165
    Progress: 69.0% ... Training loss: 0.098 ... Validation loss: 0.178
    Progress: 69.5% ... Training loss: 0.110 ... Validation loss: 0.172
    Progress: 70.0% ... Training loss: 0.113 ... Validation loss: 0.207
    Progress: 70.5% ... Training loss: 0.105 ... Validation loss: 0.181
    Progress: 71.0% ... Training loss: 0.131 ... Validation loss: 0.208
    Progress: 71.5% ... Training loss: 0.113 ... Validation loss: 0.194
    Progress: 72.0% ... Training loss: 0.095 ... Validation loss: 0.193
    Progress: 72.5% ... Training loss: 0.106 ... Validation loss: 0.183
    Progress: 73.0% ... Training loss: 0.104 ... Validation loss: 0.175
    Progress: 73.5% ... Training loss: 0.103 ... Validation loss: 0.182
    Progress: 74.0% ... Training loss: 0.105 ... Validation loss: 0.236
    Progress: 74.5% ... Training loss: 0.107 ... Validation loss: 0.215
    Progress: 75.0% ... Training loss: 0.098 ... Validation loss: 0.214
    Progress: 75.5% ... Training loss: 0.100 ... Validation loss: 0.168
    Progress: 76.0% ... Training loss: 0.095 ... Validation loss: 0.162
    Progress: 76.5% ... Training loss: 0.092 ... Validation loss: 0.176
    Progress: 77.0% ... Training loss: 0.107 ... Validation loss: 0.162
    Progress: 77.5% ... Training loss: 0.133 ... Validation loss: 0.178
    Progress: 78.0% ... Training loss: 0.100 ... Validation loss: 0.180
    Progress: 78.5% ... Training loss: 0.092 ... Validation loss: 0.181
    Progress: 79.0% ... Training loss: 0.095 ... Validation loss: 0.147
    Progress: 79.5% ... Training loss: 0.098 ... Validation loss: 0.174
    Progress: 80.0% ... Training loss: 0.097 ... Validation loss: 0.175
    Progress: 80.5% ... Training loss: 0.098 ... Validation loss: 0.154
    Progress: 81.0% ... Training loss: 0.142 ... Validation loss: 0.175
    Progress: 81.5% ... Training loss: 0.094 ... Validation loss: 0.158
    Progress: 82.0% ... Training loss: 0.090 ... Validation loss: 0.163
    Progress: 82.5% ... Training loss: 0.102 ... Validation loss: 0.174
    Progress: 83.0% ... Training loss: 0.091 ... Validation loss: 0.161
    Progress: 83.5% ... Training loss: 0.093 ... Validation loss: 0.184
    Progress: 84.0% ... Training loss: 0.099 ... Validation loss: 0.160
    Progress: 84.5% ... Training loss: 0.101 ... Validation loss: 0.171
    Progress: 85.0% ... Training loss: 0.190 ... Validation loss: 0.198
    Progress: 85.5% ... Training loss: 0.090 ... Validation loss: 0.171
    Progress: 86.0% ... Training loss: 0.125 ... Validation loss: 0.267
    Progress: 86.5% ... Training loss: 0.088 ... Validation loss: 0.153
    Progress: 87.0% ... Training loss: 0.113 ... Validation loss: 0.173
    Progress: 87.5% ... Training loss: 0.101 ... Validation loss: 0.161
    Progress: 88.0% ... Training loss: 0.089 ... Validation loss: 0.161
    Progress: 88.5% ... Training loss: 0.085 ... Validation loss: 0.149
    Progress: 89.0% ... Training loss: 0.103 ... Validation loss: 0.170
    Progress: 89.5% ... Training loss: 0.090 ... Validation loss: 0.177
    Progress: 90.0% ... Training loss: 0.091 ... Validation loss: 0.154
    Progress: 90.5% ... Training loss: 0.090 ... Validation loss: 0.155
    Progress: 91.0% ... Training loss: 0.133 ... Validation loss: 0.145
    Progress: 91.5% ... Training loss: 0.108 ... Validation loss: 0.154
    Progress: 92.0% ... Training loss: 0.104 ... Validation loss: 0.187
    Progress: 92.5% ... Training loss: 0.109 ... Validation loss: 0.175
    Progress: 93.0% ... Training loss: 0.106 ... Validation loss: 0.167
    Progress: 93.5% ... Training loss: 0.141 ... Validation loss: 0.181
    Progress: 94.0% ... Training loss: 0.092 ... Validation loss: 0.170
    Progress: 94.5% ... Training loss: 0.098 ... Validation loss: 0.164
    Progress: 95.0% ... Training loss: 0.086 ... Validation loss: 0.179
    Progress: 95.5% ... Training loss: 0.089 ... Validation loss: 0.199
    Progress: 96.0% ... Training loss: 0.090 ... Validation loss: 0.181
    Progress: 96.5% ... Training loss: 0.091 ... Validation loss: 0.162
    Progress: 97.0% ... Training loss: 0.097 ... Validation loss: 0.157
    Progress: 97.5% ... Training loss: 0.099 ... Validation loss: 0.171
    Progress: 98.0% ... Training loss: 0.082 ... Validation loss: 0.165
    Progress: 98.5% ... Training loss: 0.094 ... Validation loss: 0.176
    Progress: 99.0% ... Training loss: 0.085 ... Validation loss: 0.184
    Progress: 99.5% ... Training loss: 0.090 ... Validation loss: 0.177


```python
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=0.5)
```




    (0.0, 0.5)




![png](output_20_1.png)



```python
fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

#print(MSE(predictions[0],(test_targets['cnt']*std + mean).values))
```


![png](output_21_0.png)


## Thinking about your results
 
Answer these questions about your results. How well does the model predict the data? Where does it fail? Why does it fail where it does?

> **Note:** You can edit the text in this cell by double clicking on it. When you want to render the text, press control + enter

#### Your answer below
It looks good before Dec 21, but after Dec 22 the predictions fail. It's hard to say why does it fail, but the data show that the cnt after Dec 22 is smaller than before, may there are other features affact cnt.

## Check out your predictions

Here, use the test data to view how well your network is modeling the data. If something is completely wrong here, make sure each step in your network is implemented correctly.

## Unit tests

Run these unit tests to check the correctness of your network implementation. These tests must all be successful to pass the project.


```python
import unittest

inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3], 
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014,  0.39775194, -0.29887597],
                                              [-0.20185996,  0.50074398,  0.19962801]])))

    def runTest(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)
```

    ....
    ----------------------------------------------------------------------
    Ran 4 tests in 0.011s
    
    OK
    




    <unittest.runner.TextTestResult run=4 errors=0 failures=0>


