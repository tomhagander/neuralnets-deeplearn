import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from utils.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, customnet=False, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
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
    self.customnet = customnet
    self.bn_param1 = {'mode': 'train'}
    self.bn_param2 = {'mode': 'train'}
    self.bn_param3 = {'mode': 'train'}
    self.bn_param4 = {'mode': 'train'}
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    if customnet:
      self.params['gamma1'] = np.ones(num_filters)
      self.params['beta1'] = np.zeros(num_filters)
      self.params['gamma2'] = np.ones(num_filters)
      self.params['beta2'] = np.zeros(num_filters)
      self.params['gamma3'] = np.ones(hidden_dim)
      self.params['beta3'] = np.zeros(hidden_dim)
      self.params['gamma4'] = np.ones(num_classes)
      self.params['beta4'] = np.zeros(num_classes)

      self.params['b1'] = np.zeros((num_filters,))
      self.params['b2'] = np.zeros((num_filters,))
      self.params['b5'] = np.zeros((hidden_dim,))
      self.params['b6'] = np.zeros((num_classes,))

      self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
      self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, num_filters, filter_size, filter_size))
      self.params['W5'] = np.random.normal(loc=0.0, scale=weight_scale, size=(int(num_filters*input_dim[1]*input_dim[2]/4), hidden_dim))
      self.params['W6'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
    else:
      self.params['b1'] = np.zeros((num_filters,))
      self.params['b2'] = np.zeros((hidden_dim,))
      self.params['b3'] = np.zeros((num_classes,))

      self.params['W1'] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
      self.params['W2'] = np.random.normal(loc=0.0, scale=weight_scale, size=(int(num_filters*input_dim[1]*input_dim[2]/4), hidden_dim))
      self.params['W3'] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
      print(k, v.shape)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    if self.customnet:
      W1, b1 = self.params['W1'], self.params['b1']
      W2, b2 = self.params['W2'], self.params['b2']
      W5, b5 = self.params['W5'], self.params['b5']
      W6, b6 = self.params['W6'], self.params['b6']
    else:
      W1, b1 = self.params['W1'], self.params['b1']
      W2, b2 = self.params['W2'], self.params['b2']
      W3, b3 = self.params['W3'], self.params['b3']
    
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

    if self.customnet:
      h1, cache1 = conv_relu_forward(X, W1, b1, conv_param)
      h1_bn, cache1_bn = spatial_batchnorm_forward(h1, self.params['gamma1'], self.params['beta1'], self.bn_param1)
      h2, cache2 = conv_relu_pool_forward(h1_bn, W2, b2, conv_param, pool_param)
      h2_bn, cache2_bn = spatial_batchnorm_forward(h2, self.params['gamma2'], self.params['beta2'], self.bn_param2)
      h5, cache5 = affine_relu_forward(h2_bn, W5, b5)
      h5_bn, cache5_bn = batchnorm_forward(h5, self.params['gamma3'], self.params['beta3'], self.bn_param3)
      scores_no_bn, cache6 = affine_forward(h5_bn, W6, b6)
      scores, cache6_bn = batchnorm_forward(scores_no_bn, self.params['gamma4'], self.params['beta4'], self.bn_param4)
    else:
      h1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
      h2, cache2 = affine_relu_forward(h1, W2, b2)
      scores, cache3 = affine_forward(h2, W3, b3)

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

    if self.customnet:
      softloss, dsoftloss = softmax_loss(scores, y)
      dh5_bn, grads['gamma4'], grads['beta4'] = batchnorm_backward(dsoftloss, cache6_bn)
      dh5, grads['W6'], grads['b6'] = affine_backward(dh5_bn, cache6)
      dh4_bn, grads['gamma3'], grads['beta3'] = batchnorm_backward(dh5, cache5_bn)
      dh4, grads['W5'], grads['b5'] = affine_relu_backward(dh4_bn, cache5)
      dh1_bn, grads['gamma2'], grads['beta2'] = spatial_batchnorm_backward(dh4, cache2_bn)
      dh1, grads['W2'], grads['b2'] = conv_relu_pool_backward(dh1_bn, cache2)
      dx_bn, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dh1, cache1_bn)
      dx, grads['W1'], grads['b1'] = conv_relu_backward(dx_bn, cache1)

      grads['W1'] += self.reg*W1
      grads['W2'] += self.reg*W2
      grads['W5'] += self.reg*W5
      grads['W6'] += self.reg*W6

      loss = softloss + self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W5*W5)+np.sum(W6*W6))/2
    else:
      softloss, dsoftloss = softmax_loss(scores, y)
      dh2, grads['W3'], grads['b3'] = affine_backward(dsoftloss, cache3)
      dh1, grads['W2'], grads['b2'] = affine_relu_backward(dh2, cache2)
      dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dh1, cache1)

      grads['W1'] += self.reg*W1
      grads['W2'] += self.reg*W2
      grads['W3'] += self.reg*W3

      loss = softloss + self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))/2

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  pass