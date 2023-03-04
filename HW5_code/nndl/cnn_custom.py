from locale import dcgettext
import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

class CustomConvNet(object):
  """
  A custom convolutional network with the following architecture:
  
  (conv - relu - conv - relu - 2x2 max pool)xN - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    self.params['b1'] = np.zeros((num_filters,))
    self.params['b2'] = np.zeros((num_filters,))
    self.params['b3'] = np.zeros((num_filters,))
    self.params['b4'] = np.zeros((num_filters,))
    self.params['b5'] = np.zeros((hidden_dim,))
    self.params['b6'] = np.zeros((num_classes,))

    self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
    self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size))
    self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size))
    self.params['W4'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size))
    self.params['W5'] = np.random.normal(loc=0.0, scale=weight_scale, size=(int(num_filters*input_dim[1]*input_dim[2]/4), hidden_dim))
    self.params['W6'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the custom convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    
    h1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
    h2, cache2 = conv_relu_pool_forward(h1, W2, b2, conv_param, pool_param)
    h3, cache3 = conv_relu_forward(h2, W3, b3, conv_param)
    h4, cache4 = conv_relu_pool_forward(h3, W4, b4, conv_param, pool_param)
    h5, cache5 = affine_relu_forward(h4, W5, b5)
    scores, cache6 = affine_forward(h5, W6, b6)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    softloss, dsoftloss = softmax_loss(scores, y)
    dh5, grads['W6'], grads['b6'] = affine_backward(dsoftloss, cache6)
    dh4, grads['W5'], grads['b4'] = affine_relu_backward(dh5, cache5)
    dh3, grads['W4'], grads['b4'] = conv_relu_pool_backward(dh4, cache4)
    dh2, grads['W3'], grads['b3'] = conv_relu_backward(dh3, cache3)
    dh1, grads['W2'], grads['b2'] = conv_relu_pool_backward(dh2, cache2)
    dx, grads['W1'], grads['b1'] = conv_relu_backward(dh1, cache1)

    grads['W1'] += self.reg*W1
    grads['W2'] += self.reg*W2
    grads['W3'] += self.reg*W3
    grads['W4'] += self.reg*W4
    grads['W5'] += self.reg*W5
    grads['W6'] += self.reg*W6

    loss = softloss + self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)+np.sum(W4*W4)+np.sum(W5*W5)+np.sum(W6*W6))/2

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads