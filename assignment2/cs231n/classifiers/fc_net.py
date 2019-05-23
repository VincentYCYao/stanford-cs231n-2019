from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros((1, hidden_dim))
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros((1, num_classes))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden, hidden_cache = affine_relu_forward(X, W1, b1)
        scores, scores_cache = affine_forward(hidden, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # calculate loss
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1*W1) + 0.5 * self.reg * np.sum(W2*W2)
        loss = data_loss + reg_loss

        # backpropagation 1: the output layer (second layer) (affine)
        dhidden, dW2, db2 = affine_backward(dscores, scores_cache)
        dW2 += self.reg * W2  # add regularization term

        # backpropagation 2: the hidden layer (first layer) (affine - ReLU)
        _, dW1, db1 = affine_relu_backward(dhidden, hidden_cache)
        dW1 += self.reg * W1  # add regularization term

        # save gradients
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # initialize W and b
        input_plus_layers_dim = hidden_dims.copy()
        input_plus_layers_dim.append(num_classes)
        input_plus_layers_dim.insert(0, input_dim)
        for i_layer in range(self.num_layers):
            num_rows = input_plus_layers_dim[i_layer]
            num_cols = input_plus_layers_dim[i_layer + 1]
            layer_id = i_layer + 1
            self.params['W' + str(layer_id)] = weight_scale * np.random.randn(num_rows, num_cols)
            self.params['b' + str(layer_id)] = np.zeros(num_cols)

        # initialize gamma and beta for normalization
        if self.normalization is not None:
            for i_hidden in range(len(hidden_dims)):
                layer_id = i_hidden + 1
                norm_dim = hidden_dims[i_hidden]
                self.params['gamma' + str(layer_id)] = np.ones(norm_dim)
                self.params['beta' + str(layer_id)] = np.zeros(norm_dim)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            for i in range(self.num_layers - 1):
                self.bn_params.append({'mode': 'train'})
        if self.normalization == 'layernorm':
            for i in range(self.num_layers - 1):
                self.bn_params.append({})

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                # mode is initialized as "train", change the mode in self.bn_params here
                bn_param['mode'] = mode
        scores = None

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        num_hidden = self.num_layers - 1
        cache = {}  # cache['1'] is cache for layer 1
        out = {}  # out['1'] is output of layer 1
        derivative = {}  # derivative['1'] is the gradient of out['1']

        # forward pass of hidden layers: {affine - [batch/layer norm] - relu - [dropout]}
        for i_hidden in range(num_hidden):
            if i_hidden == 0:
                x = X
            else:
                x = out[str(i_hidden)]
            layer_id = i_hidden + 1
            w = self.params['W' + str(layer_id)]
            b = self.params['b' + str(layer_id)]

            # affine - [batch/layer norm] - relu
            if self.normalization is None:
                out[str(layer_id)], cache[str(layer_id)] = affine_relu_forward(x, w, b)
            elif self.normalization == 'batchnorm':
                gamma = self.params['gamma' + str(layer_id)]
                beta = self.params['beta' + str(layer_id)]
                bn_param = self.bn_params[i_hidden]

                # NOTE: this function also change self.bn_params[i_hidden]
                #       bn_param is pointer to self.bn_params[i_hidden]
                #       this mechanism is helpful to store running_mean and running_var
                out[str(layer_id)], cache[str(layer_id)] = affine_bn_relu_forward(
                                                           x, w, b, gamma, beta, bn_param)
            elif self.normalization == 'layernorm':
                gamma = self.params['gamma' + str(layer_id)]
                beta = self.params['beta' + str(layer_id)]
                bn_param = self.bn_params[i_hidden]
                out[str(layer_id)], cache[str(layer_id)] = affine_ln_relu_forward(
                    x, w, b, gamma, beta, bn_param)
            else:
                raise ValueError('Unrecognized arguments %s' % self.normalization)

            # [dropout]
            if self.use_dropout:
                key = 'drop' + str(layer_id)
                out[str(layer_id)], cache[key] = dropout_forward(out[str(layer_id)], self.dropout_param)

        # forward pass of the last layers: {affine}
        x = out[str(num_hidden)]  # output of the last hidden layer
        w = self.params['W' + str(self.num_layers)]  # W of the last layer
        b = self.params['b' + str(self.num_layers)]  # b of the last layer
        out[str(self.num_layers)], cache[str(self.num_layers)] = affine_forward(x, w, b)
        scores = out[str(self.num_layers)]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # loss
        data_loss, derivative[str(self.num_layers)] = softmax_loss(scores, y)
        reg_loss = 0
        for i_layer in range(self.num_layers):
            layer_id = i_layer + 1
            reg_loss += 0.5 * self.reg * np.sum(self.params['W' + str(layer_id)]**2)
        loss = data_loss + reg_loss

        # backward pass of the last layers: {affine}
        dx, dw, db = affine_backward(derivative[str(self.num_layers)], cache[str(self.num_layers)])
        _, w, _ = cache[str(self.num_layers)]
        dw += self.reg * w
        grads['W' + str(self.num_layers)] = dw
        grads['b' + str(self.num_layers)] = db
        derivative[str(self.num_layers - 1)] = dx

        # backward pass of hidden layers: {affine - [batch/layer norm] - relu - [dropout]}
        for i_hidden in reversed(range(num_hidden)):
            layer_id = i_hidden + 1

            # [dropout]
            if self.use_dropout:
                key = 'drop' + str(layer_id)
                drop_cache = cache[key]
                ddrop = derivative[str(layer_id)]
                derivative[str(layer_id)] = dropout_backward(ddrop, drop_cache)

            # affine - [batch/layer norm] - relu
            if self.normalization is None:
                dx, dw, db = affine_relu_backward(derivative[str(layer_id)], cache[str(layer_id)])
                fc_cache, _ = cache[str(layer_id)]
                _, w, _ = fc_cache
                dw += self.reg * w
                grads['W' + str(layer_id)] = dw
                grads['b' + str(layer_id)] = db
                derivative[str(layer_id - 1)] = dx
            elif self.normalization == 'batchnorm':
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(derivative[str(layer_id)],
                                                                    cache[str(layer_id)])
                fc_cache, _, _ = cache[str(layer_id)]
                _, w, _ = fc_cache
                dw += self.reg * w
                grads['W' + str(layer_id)] = dw
                grads['b' + str(layer_id)] = db
                grads['gamma' + str(layer_id)] = dgamma
                grads['beta' + str(layer_id)] = dbeta
                derivative[str(layer_id - 1)] = dx
            elif self.normalization == 'layernorm':
                dx, dw, db, dgamma, dbeta = affine_ln_relu_backward(derivative[str(layer_id)],
                                                                    cache[str(layer_id)])
                fc_cache, _, _ = cache[str(layer_id)]
                _, w, _ = fc_cache
                dw += self.reg * w
                grads['W' + str(layer_id)] = dw
                grads['b' + str(layer_id)] = db
                grads['gamma' + str(layer_id)] = dgamma
                grads['beta' + str(layer_id)] = dbeta
                derivative[str(layer_id - 1)] = dx
            else:
                raise ValueError('Unrecognized arguments %s' % self.normalization)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
