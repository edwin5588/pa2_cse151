################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import numpy as np
import math
from utils import *
from train import *
from main import *

class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)

        : Sabrina
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(
                "%s is not implemented." % (activation_type))

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        self.x = x
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = x
        return np.maximum(0, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - (self.tanh(self.x) ** 2)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return 1 * (self.x > 0)


class Layer:
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(1024, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)

    : Edwin
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = math.sqrt(2 / in_units) * np.random.randn(in_units,
                                                           out_units)  # You can experiment with initialization.
        self.b = np.zeros((1, out_units))  # Create a placeholder for Bias
        self.x = None  # Save the input to forward in this
        # Save the output of forward pass in this (without activation)
        self.a = None

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        self.prev_increment = np.zeros((in_units, out_units))
        self.prev_bias_inc = np.zeros((1, out_units))

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = (self.x @ self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.d_x
        """
        # gradient equation: wij = wij + lr * deltaj * xi + bias
        # deltaj = slope * deltak*wjk
        # print('delta', delta.shape)
        # print(self.w.shape,
        #       self.b.shape,
        #       self.x.shape,
        #       self.a.shape)

        ####### Calculate delta to pass down to next layer ##########
        self.d_w = self.x.T @ delta
        self.d_x = delta @ self.w.T

        ones = np.ones(self.x.T.shape[1])
        self.d_b = ones @ delta

        return self.d_x

    def update_weights(self, config):
        """
        Update weights for this layer based on deltas calculated
        """
        # equations for the momentum method
        # effect of the gradient is to increment the previous average
        # average also decays by alpha, slightly less than 1
        # velocity: v(t) = gamma * v(t - 1) - error * d_w(t)
        # delta w(t) = v(t)

        if config['momentum']:
            mt=config['momentum_gamma']
        else:
            mt=0

        self.prev_increment = (mt * self.prev_increment +  self.d_w)
        #  L2 stuff: - (2 * config['L2_penalty'] * self.w)

        self.w += (config['learning_rate'] / config['batch_size']) * self.prev_increment

        # bias velocity, with delta bias * learning rate
        self.prev_bias_inc = (mt * self.prev_bias_inc + self.d_b)
        # L2 Stuff: - (2 * config['L2_penalty'] * self.b)
        self.b += (config['learning_rate'] / config['batch_size']) * self.prev_bias_inc

class NeuralNetwork:
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()

    : Andrew
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers=[]  # Store all layers in this list.
        self.x=None  # Save the input to forward in this
        self.y=None  # Save the output vector of model in this
        self.targets=None  # Save the targets in forward in this variable
        self.config=config

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(
                Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """

        self.x=x
        self.targets=targets
        output=None
        layer_num=0
        
        while layer_num < len(self.layers):
            # input layer
            if layer_num == 0:
                output=self.layers[layer_num](x)
                # go to activation layer
                layer_num += 1
                output=self.layers[layer_num](output)
            else:
                # regular layer
                output=self.layers[layer_num](output)
                # go to activation layer if it is not the last layer
                if layer_num < len(self.layers) - 1:
                    layer_num += 1
                    output=self.layers[layer_num](output)

            layer_num += 1

        output=self.softmax(output)
        
        self.y=output

        if targets is not None:
            loss=self.loss(output, targets)
            return output, loss
        else:
            return output

    def backward(self):
        """
        Implement backpropagation here.
        Call backward methods of individual layer's.
        """
        layer_num=len(self.layers) - 1
        back_grad=None
        while layer_num > -1:
            # output layer
            if layer_num == len(self.layers) - 1:
                back_grad=self.layers[layer_num].backward(self.targets - self.y)
                # layer_num -= 1
            else:
                # for remaining layer
                # delta_i = self.layers[layer_num].backward(back_grad)
                back_grad = self.layers[layer_num].backward(back_grad)
            layer_num -= 1

        # Update weights after computing all deltas
        for layer in self.layers:
            if type(layer) is Layer:
                layer.update_weights(self.config)

    def softmax(self, x):
        """
        Implement the softmax function here.
        Remember to take care of the overflow condition.
        """
        if np.isnan(x).any():
            print('Very Bad')
            exit()
        x = x - np.max(x)
        return np.exp(x) / np.atleast_2d(np.sum(np.exp(x), axis=1)).T

    def loss(self, logits, targets):
        """
        compute the categorical cross-entropy loss and return it.
        """
        e = 1.0e-7

        loss= -np.mean(targets * np.log(logits + e))
        
        return loss

def approx_error(model, X_sample, y_sample, epsilon, config):

    originals = np.array([
        # weight, input to hidden, 10 to 6
        model.layers[0].w[10][6].copy(),
        # weight, input to hidden, 6 to 7
        model.layers[0].w[6][7].copy(),
        # output layer weight, the weight going from hidden 13 to output 9
        model.layers[-1].w[13][9].copy(),
        # weight, hidden to output, 5 to 8
        model.layers[-1].w[5][8].copy(),

        # output bias weight
        model.layers[-1].b[0][9].copy(),
        # hidden bias weight
        model.layers[0].b[0][6].copy()
    ])

    approx = np.empty(6)

    ################################### 1 ####################################

    model.layers[0].w[10][6] = originals[0]
    # E(w+e)
    model.layers[0].w[10][6] += epsilon
    # perform forward pass for one training example and compute the loss
    something, big_loss = model.forward(X_sample, targets = y_sample)

    model.layers[0].w[10][6] = originals[0]
    #E(w-e)
    model.layers[0].w[10][6] -= epsilon
    # perform forward pass for one training example and compute the loss
    something, small_loss = model.forward(X_sample, targets = y_sample)
    model.layers[0].w[10][6] = originals[0]
    # model.backward()
    # original_gradients[0] = model.layers[0].d_w[10][6].copy()
    approx[0] = (big_loss - small_loss)/(2 * epsilon)

################################### 2 ####################################

    model.layers[0].w[6][7] = originals[1]
    # E(w+e)
    model.layers[0].w[6][7] += epsilon
    # perform forward pass for one training example and compute the loss
    something, big_loss = model.forward(X_sample, targets = y_sample)

    model.layers[0].w[6][7] = originals[1]
    #E(w-e)
    model.layers[0].w[6][7] -= epsilon
    # perform forward pass for one training example and compute the loss
    something, small_loss = model.forward(X_sample, targets = y_sample)

    model.layers[0].w[6][7] = originals[1]
    # model.backward()
    # original_gradients[1] = model.layers[0].d_w[6][7].copy()
    approx[1] = (big_loss - small_loss)/(2 * epsilon)

################################### 3 ####################################

    model.layers[-1].w[13][9] = originals[2]
    # E(w+e)
    model.layers[-1].w[13][9] += epsilon
    # perform forward pass for one training example and compute the loss
    something, big_loss = model.forward(X_sample, targets = y_sample)

    model.layers[-1].w[13][9] = originals[2]
    #E(w-e)
    model.layers[-1].w[13][9] -= epsilon
    # perform forward pass for one training example and compute the loss
    something, small_loss = model.forward(X_sample, targets = y_sample)

    model.layers[-1].w[13][9] = originals[2]
    # model.backward()
    # original_gradients[2] = model.layers[-1].d_w[13][9].copy()
    approx[2] = (big_loss - small_loss)/(2 * epsilon)


################################### 4 ####################################

    model.layers[-1].w[5][8] = originals[3]
    # E(w+e)
    model.layers[-1].w[5][8] += epsilon
    # perform forward pass for one training example and compute the loss
    something, big_loss = model.forward(X_sample, targets = y_sample)

    model.layers[-1].w[5][8] = originals[3]
    #E(w-e)
    model.layers[-1].w[5][8] -= epsilon
    # perform forward pass for one training example and compute the loss
    something, small_loss = model.forward(X_sample, targets = y_sample)

    model.layers[-1].w[5][8] = originals[3]
    # model.backward()
    # original_gradients[3] = model.layers[-1].d_w[5][8].copy()
    approx[3] = (big_loss - small_loss)/(2 * epsilon)


################################### 5 ####################################

    model.layers[-1].b[0][9] = originals[4]
    # E(w+e)
    model.layers[-1].b[0][9] += epsilon
    # perform forward pass for one training example and compute the loss
    something, big_loss = model.forward(X_sample, targets = y_sample)


    model.layers[-1].b[0][9] = originals[4]
    #E(w-e)
    model.layers[-1].b[0][9] -= epsilon
    # perform forward pass for one training example and compute the loss
    something, small_loss = model.forward(X_sample, targets = y_sample)

    model.layers[-1].b[0][9] = originals[4]

    approx[4] = (big_loss - small_loss)/(2 * epsilon)


################################### 6 ####################################

    model.layers[0].b[0][6] = originals[5]
    # E(w+e)
    model.layers[0].b[0][6] += epsilon
    # perform forward pass for one training example and compute the loss
    something, big_loss = model.forward(X_sample, targets = y_sample)

    model.layers[0].b[0][6] = originals[5]
    #E(w-e)
    model.layers[0].b[0][6] -= epsilon
    # perform forward pass for one training example and compute the loss
    something, small_loss = model.forward(X_sample, targets = y_sample)

    model.layers[0].b[0][6] = originals[5]

    approx[5] = (big_loss - small_loss)/(2 * epsilon)

    model.forward(X_sample, targets= y_sample)
    model.backward()
    original_gradients = np.array([
        # weight, input to hidden, 10 to 6
        model.layers[0].d_w[10][6].copy(),
        # weight, input to hidden, 6 to 7
        model.layers[0].d_w[6][7].copy(),
        # output layer weight, the weight going from hidden 13 to output 9
        model.layers[-1].d_w[13][9].copy(),
        # weight, hidden to output, 5 to 8
        model.layers[-1].d_w[5][8].copy(),

        # output bias weight
        model.layers[-1].d_b[9].copy(),
        # hidden bias weight
        model.layers[0].d_b[6].copy()
    ])

    print(big_loss, "<-- big and small -->",small_loss, "\n")

    print("approximations: ", approx/config['batch_size'])
    print("backprop_gradients: ", original_gradients/config['batch_size'])
    print("difference: ", (approx/config['batch_size'] - original_gradients/config['batch_size']))



if __name__ == "__main__":
    config = load_config("./config.yaml")

    # Load the data
    x_train, y_train, x_test, y_test = load_data()
    print('Data Shape:', x_train.shape,
          y_train.shape, x_test.shape, y_test.shape)
    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)

    # Create validation set out of training data.
    ind = np.arange(len(x_train))
    np.random.shuffle(ind)
    eighty = int(np.ceil(len(x_train) * 0.8))

    x_train, y_train = x_train[ind], y_train[ind]


    x_train, y_train, x_val, y_val = x_train[0:eighty], y_train[0:eighty], x_train[eighty:], y_train[eighty:]

    # Any pre-processing on the datasets goes here.
    # vectorize the datasets
    x_train = np.array([img_2_vec(a) for a in x_train])
    x_val = np.array([img_2_vec(a) for a in x_val])
    x_test = np.array([img_2_vec(a) for a in x_test])

    # normalize everything
    x_train, x_val = normalize_z(x_train), normalize_z(x_val)
    x_test = normalize_z(x_test)

    print('Preprocess data shape: ', x_train.shape, y_train.shape, x_val.shape,
          y_val.shape, x_test.shape, y_test.shape)

    nn = NeuralNetwork(config)


    approx_error(nn, x_train[0][np.newaxis, :], y_train[0][np.newaxis, :], 10e-2,config)

    e = 1.0e-2
