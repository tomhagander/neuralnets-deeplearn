import numpy as np


class Softmax(object):


  def __init__(self, dims=[10, 3073]):
    self.init_weights(dims=dims)

  def init_weights(self, dims):
    """
    Initializes the weight matrix of the Softmax classifier.  
    Note that it has shape (C, D) where C is the number of 
    classes and D is the feature size.
    """
    self.W = np.random.normal(size=dims) * 0.0001

  def loss(self, X, y):
    """
    Calculates the softmax loss.
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
  
    Returns a tuple of:
    - loss as single float
    """

    # Initialize the loss to zero.
    loss = 0.0

    N = X.shape[0]
    C, D = self.W.shape

    # loss = 1/N * sum over examples (ln (sum over classes j exp(wjTxi)) - w(y(i))Txi)

    sum = 0
    for i in range(N):
      sum += np.log(np.sum(np.exp(self.W.dot(X[i])))) - self.W[y[i]].dot(X[i])

    loss = sum/N 

    return loss

  def loss_and_grad(self, X, y):
    """
    Same as self.loss(X, y), except that it also returns the gradient.

    Output: grad -- a matrix of the same dimensions as W containing 
      the gradient of the loss with respect to W.
    """

    N = X.shape[0]
    C, D = self.W.shape
    # Initialize the loss and gradient to zero.
    loss = self.loss(X, y)
    grad = np.zeros_like(self.W)

    for j in range(C):
      sum = np.zeros_like(grad[j])
      for i in range(N):
        sum += X[i]*np.exp(self.W[j].dot(X[i]))/(np.sum(np.exp(self.W.dot(X[i]))))
        if y[i] == j:
          sum -= X[i]

      grad[j] = sum/N
      

    return loss, grad

  def grad_check_sparse(self, X, y, your_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in these dimensions.
    """
  
    for i in np.arange(num_checks):
      ix = tuple([np.random.randint(m) for m in self.W.shape])
  
      oldval = self.W[ix]
      self.W[ix] = oldval + h # increment by h
      fxph = self.loss(X, y)
      self.W[ix] = oldval - h # decrement by h
      fxmh = self.loss(X,y) # evaluate f(x - h)
      self.W[ix] = oldval # reset
  
      grad_numerical = (fxph - fxmh) / (2 * h)
      grad_analytic = your_grad[ix]
      rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
      print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

  def fast_loss_and_grad(self, X, y):
    """
    A vectorized implementation of loss_and_grad. It shares the same
    inputs and ouptuts as loss_and_grad.
    """

    N = X.shape[0]
    C, D = self.W.shape

    #loss
    indexlist = np.array([np.arange(len(y)), y])
    loss = (1/N)*np.sum(np.log(np.sum(np.exp(X.dot(self.W.T)),axis=1)),axis=0) - (1/N)*np.sum((X.dot(self.W.T))[indexlist.T])

    #grad
    Ys = np.tile(y, (C,1))
    indices = np.vstack([np.arange(C),]*N).T
    indicator = np.equal(Ys, indices).astype(int)
    coeff = np.divide(np.exp(self.W.dot(X.T)), np.sum(np.exp(self.W.dot(X.T)), axis=0))

    grad = (coeff - indicator).dot(X)/N 

    # grad = np.zeros_like(self.W)

    return loss, grad

  def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes

    self.init_weights(dims=[np.max(y) + 1, X.shape[1]])	# initializes the weights of self.W

    # Run stochastic gradient descent to optimize W
    loss_history = []

    for it in np.arange(num_iters):

      indices = np.random.choice(np.arange(len(y)), batch_size)
      X_batch = X[indices]
      y_batch = y[indices]

      # evaluate loss and gradient
      loss, grad = self.fast_loss_and_grad(X_batch, y_batch)
      loss_history.append(loss)

      self.W -= learning_rate*grad

      if verbose and it % 100 == 0:
        print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Inputs:
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])

    for i in range(len(y_pred)):
      preds = self.W.dot(X[i].T)
      y_pred[i] = np.argmax(preds)

    return y_pred

