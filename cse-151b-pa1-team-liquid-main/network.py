import numpy as np
import data
import time
import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    
    return 1 / (1 + np.exp(-a))


def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    re = []
    for x in a:
        row = np.exp(x) / np.sum(np.exp(x))
        re.append(row)
    return re

def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    
    return -(t * np.log(y) + (1-t) * np.log(1-y))

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    idx = np.argmax(t)
    return -np.exp(y[idx])

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss
        self.out_dim = out_dim

        self.weights = np.zeros((32*32+1, out_dim))

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        #a = np.dot(X,self.weights)
        a = X@self.weights # X.shape=(n,1025), weights.shape=(1025,out_dim)
        out = self.activation(a) # a.shape=(n,out_idm), out.shape=(n,out_dim)
        return out

    def __call__(self, X):
        return self.forward(X)

    def train(self, minibatch):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        X, y = minibatch[0], minibatch[1]
        X = data.append_bias(X)
        y = np.reshape(y,(len(y),self.out_dim))
        pred = np.reshape(self.forward(X),(len(y),self.out_dim))
        grad = np.zeros((32*32+1,self.out_dim))
        loss = 0
        accuracy = 0
        for i in range(len(y)):
            t_y = [(y[i][it] - pred[i][it]) for it in range(self.out_dim)]
            grad += -np.reshape(np.array(X[i]),(32*32+1,1))@np.reshape(t_y,(1,self.out_dim))
            #grad += -(y[i] - pred[i])*X[i] #X[i] is a vector with length 32*32+1, correspond to weight vector
            #print("cross loss:",self.loss(pred[i],y[i]))
            #print("pred[i]",pred[i])
            loss += self.loss(pred[i],y[i])
            idx_p = np.argmax(pred[i])
            idx_y = np.argmax(y[i])
            if idx_p == idx_y: # this should be 0.5 right?
                if len(pred[i]) == 1:
                    if pred[i][0] > 0.5 and y[i][0] == 1:
                        accuracy += 1
                    elif pred[i][0] <= 0.5 and y[i][0] == 0:
                        accuracy += 1
                else:
                    accuracy += 1
        alpha = self.hyperparameters[0]
        for i in range(len(self.weights)):
            self.weights[i] += alpha * grad[i]
        return (loss/len(y), accuracy/len(y))

    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        X, y = minibatch[0], minibatch[1]
        X = data.append_bias(X)
        y = np.reshape(y,(len(y),self.out_dim))
        pred = np.reshape(self.forward(X),(len(y),self.out_dim))
        loss = 0
        accuracy = 0
        for i in range(len(y)):
            loss += self.loss(pred[i],y[i])
            idx_p = np.argmax(pred[i])
            idx_y = np.argmax(y[i])
            if idx_p == idx_y:
                if len(pred[i]) == 1:
                    if pred[i][0] > 0.5 and y[i][0] == 1:
                        accuracy += 1
                    elif pred[i][0] <= 0.5 and y[i][0] == 0:
                        accuracy += 1
                else:
                    accuracy += 1
        return (loss/len(y),accuracy/len(y))
