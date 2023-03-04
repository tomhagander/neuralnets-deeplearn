from xml.sax.handler import property_declaration_handler
import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H_prime = int(1 + (H + 2*pad - HH)/stride)
  W_prime = int(1 + (W + 2*pad - WW)/stride)
  out = np.zeros(shape=(N, F, H_prime, W_prime))
  x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)))
  
  for n in range(N):
    # elementwise multiplication of two matrices, then sum
    for r in range(H_prime):
      for c in range(W_prime):
        for f in range(F):
          filter = w[f,:,:,:]
          img_part = x_padded[n,:,stride*r:(stride*r + HH), stride*c:(stride*c + WW)]
          out[n,f,r,c] = np.multiply(filter, img_part).sum() + b[f]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  db = np.zeros_like(b)
  dx_padded = np.zeros_like(xpad)
  dw = np.zeros_like(w)
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H_prime = int(1 + (H + 2*pad - HH)/stride)
  W_prime = int(1 + (W + 2*pad - WW)/stride)

  for n in range(N):
    for f in range(F):
      db[f] += dout[n,f,:,:].sum()
      for r in range(H_prime):
        for c in range(W_prime):
          dw[f, :, :, :] += xpad[n, :, stride*r:(stride*r + HH), stride*c:(stride*c + WW)]*dout[n,f,r,c]
          dx_padded[n, :, stride*r:(stride*r + HH), stride*c:(stride*c + WW)] += dout[n,f,r,c]*w[f]


  dx = dx_padded[:, :, pad:-pad, pad:-pad]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  # out shape is N, C, (1 + (H - pool_height)/stride), (1 + (W - pool_width)/stride)
  N, C, H, W = x.shape
  stride = pool_param['stride']
  pool_H = pool_param['pool_height']
  pool_W = pool_param['pool_width']
  Hout = int(1 + (H - pool_H)/stride)
  Wout = int(1 + (W - pool_W)/stride)
  out = np.zeros((N, C, Hout, Wout))

  for n in range(N):
    for channel in range(C):
      for r in range(Hout):
        for c in range(Wout):
          out[n,channel,r,c] = np.max(x[n,channel,stride*r:stride*r + pool_H, stride*c:stride*c + pool_W])

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  stride = pool_param['stride']
  pool_H = pool_param['pool_height']
  pool_W = pool_param['pool_width']
  Hout = int(1 + (H - pool_H)/stride)
  Wout = int(1 + (W - pool_W)/stride)
  dx = np.zeros_like(x)

  for n in range(N):
    for channel in range(C):
      for r in range(Hout):
        for c in range(Wout):
          sq = x[n,channel,stride*r:stride*r + pool_H, stride*c:stride*c + pool_W]
          max_val = np.max(sq)
          mask = (sq == max_val)
          dx[n,channel,stride*r:stride*r + pool_H, stride*c:stride*c + pool_W] += mask*dout[n,channel,r,c]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = x.shape
  temp = np.transpose(x, (0, 2, 3, 1))
  reshaped = np.reshape(temp, (N*H*W, C))
  from nndl.layers import batchnorm_forward
  tempout, cache = batchnorm_forward(reshaped, gamma, beta, bn_param)
  out_no_transpose = np.reshape(tempout, (N,H,W,C))
  out = np.transpose(out_no_transpose, (0,3,1,2))


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = dout.shape
  temp = np.transpose(dout, (0, 2, 3, 1))
  reshaped = np.reshape(temp, (N*H*W, C))
  from nndl.layers import batchnorm_forward
  from nndl.layers import batchnorm_backward
  temp_dx, dgamma, dbeta = batchnorm_backward(reshaped, cache)
  dx_no_transpose = np.reshape(temp_dx, (N,H,W,C))
  dx = np.transpose(dx_no_transpose, (0,3,1,2))


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta