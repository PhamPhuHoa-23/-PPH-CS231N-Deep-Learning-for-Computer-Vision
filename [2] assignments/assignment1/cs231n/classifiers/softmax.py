from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    linear_predict = X @ W

    # Normalize result for further calculation
    linear_predict = linear_predict - linear_predict.max(axis=1, keepdims=True)

    # Get the exponential result
    exponential_result = np.exp(linear_predict)

    # Normalize result
    softmax_matrix = exponential_result / exponential_result.sum(axis=1, keepdims=True)

    # Calculating loss
    loss = -np.log(softmax_matrix[range(len(y)), y]).mean()

    loss += reg * np.sum(W * W)

    # Add 1 for true label
    softmax_matrix[range(len(y)), y] = softmax_matrix[range(len(y)), y] - 1

    dW = X.T @ softmax_matrix / len(y) + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    linear_predict = X @ W

    # Normalize result for further calculation
    linear_predict = linear_predict - linear_predict.max(axis=1, keepdims=True)

    # Get the exponential result
    exponential_result = np.exp(linear_predict)

    # Normalize result
    softmax_matrix = exponential_result / exponential_result.sum(axis=1, keepdims=True)

    # Calculating loss
    loss = -np.log(softmax_matrix[range(len(y)), y]).mean()

    loss += reg * np.sum(W * W)

    # Add 1 for true label
    softmax_matrix[range(len(y)), y] = softmax_matrix[range(len(y)), y] - 1

    dW = X.T @ softmax_matrix / len(y) + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
