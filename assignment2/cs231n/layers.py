from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_2d = x.reshape(x.shape[0], -1)  # reshape x to size [N, D], D for 'dimension'
    out = np.dot(x_2d, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_2d = x.reshape(x.shape[0], -1)  # reshape x to size [N, D], D for 'dimension'
    dx = np.dot(dout, w.T).reshape(x.shape)  # return to same shape as x
    dw = np.dot(x_2d.T, dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.zeros_like(x)
    dx[x > 0] = dout[x > 0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, {}
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # batchnorm forward pass
        mean = np.mean(x, axis=0)  # [D,]
        e = x - mean  # error: [N, D]
        var = np.mean(e**2, axis=0)  # variance: [D,]
        std = np.sqrt(var + eps)  # standard deviation: [D,]
        invstd = 1 / std  # inverted standard deviation: [D,]
        xhat = e * invstd  # normalized x: [N, D]
        out = gamma * xhat + beta  # scaled and shifted normalized x: [N, D]

        # cache
        cache = {'e': e, 'var': var, 'std': std, 'invstd': invstd,
                 'xhat': xhat, 'gamma': gamma, 'N': N, 'D': D, 'eps': eps}

        # update running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        norm_x = (x - running_mean) / np.sqrt(running_var)
        out = norm_x * gamma + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # unpack cache
    e = cache['e']  # [N, D]
    var = cache['var']  # variance: [D,]
    std = cache['std']  # standard deviation: [D,]
    invstd = cache['invstd']  # inverted standard deviation: [D,]
    xhat = cache['xhat']  # normalized x: [N, D]
    gamma = cache['gamma']  # [D,]
    N = cache['N']  # number of subjects
    D = cache['D']  # dimension (number of features)
    eps = cache['eps']

    # backpropogation for dgamma and dbeta
    dbeta = np.sum(dout, axis=0)  # [D,]
    dgamma_xhat = dout  # [N, D]
    dgamma = np.sum(dgamma_xhat * xhat, axis=0)  # [D,]

    # backprop for dx (version1)
    dxhat = gamma * dgamma_xhat  # [N, D]
    dinvstd = np.sum(e * dxhat, axis=0)  # [D,]
    dstd = -1 / std**2 * dinvstd  # [D,]
    dvar = 0.5 / np.sqrt(var + eps) * dstd  # [D,] NOTE: don't forget eps
    dse = 1/N * np.ones((N, D)) * dvar  # [N, D]
    de = 2 * e * dse  # [N, D]
    de += invstd * dxhat  # [N, D]
    dx = de  # [N, D]
    dmean = -1 * np.sum(de, axis=0)  # [D,]
    dx += 1/N * np.ones((N, D)) * dmean  # [N, D]

    # backprop for dx (version2)
    # dxhat = gamma * dgamma_xhat  # [N, D]
    # dinvstd = np.sum(e * dxhat, axis=0)  # [D,]
    # dstd = -1 / std ** 2 * dinvstd  # [D,]
    # dvar = 0.5 / var ** 0.5 * dstd  # [D,]
    # # this is the same as 'dmean' calculated from 'de' above
    # dmean = np.sum(dxhat * -invstd, axis=0) + dvar * np.mean(-2 * e, axis=0)  # [D,]
    # # the first part is actually 'de'
    # dx = (dxhat * invstd + 2 * e * dvar / N) + dmean / N  # [N, D]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # unpack cache
    e = cache['e']  # [N, D]
    std = cache['std']  # standard deviation: [D,]
    invstd = cache['invstd']  # inverted standard deviation: [D,]
    xhat = cache['xhat']  # normalized x: [N, D]
    gamma = cache['gamma']  # [D,]
    N = cache['N']  # number of subjects
    D = cache['D']  # dimension (number of features)

    # backpropogation for dgamma and dbeta
    dbeta = np.sum(dout, axis=0)  # [D,]
    dgamma = np.sum(dout * xhat, axis=0)  # [D,]

    # backpropogation for dx
    de = -1/N * e * np.sum(e * gamma * dout, axis=0) * std**(-3) + invstd * gamma * dout  # [N, D]
    dx = de - 1/N * np.ones((N, D)) * np.sum(de, axis=0)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = x.shape

    # layer normalization: forward pass
    mean = x.mean(axis=1).reshape((-1, 1))  # [N, 1]
    e = x - mean  # error: [N, D]
    var = np.mean(e**2, axis=1).reshape((-1, 1))  # variance: [N, 1]
    std = np.sqrt(var + eps)  # standard deviation: [N, 1]
    invstd = 1 / std  # inverted standard deviation: [N, 1]
    xhat = e * invstd  # normalized x: [N, D]
    out = gamma * xhat + beta  # scaled and shifted normalized x: [N, D]

    # cache
    cache = {'e': e, 'var': var, 'std': std, 'invstd': invstd,
             'xhat': xhat, 'gamma': gamma, 'N': N, 'D': D, 'eps': eps}

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # unpack cache
    e = cache['e']  # [N, D]
    var = cache['var']  # variance: [N, 1]
    std = cache['std']  # standard deviation: [N, 1]
    invstd = cache['invstd']  # inverted standard deviation: [N, 1]
    xhat = cache['xhat']  # normalized x: [N, D]
    gamma = cache['gamma']  # [D,]
    N = cache['N']  # number of subjects
    D = cache['D']  # dimension (number of features)
    eps = cache['eps']

    # backpropogation for dgamma and dbeta
    dbeta = np.sum(dout, axis=0)  # [D,]
    dgamma_xhat = dout  # [N, D]
    dgamma = np.sum(dgamma_xhat * xhat, axis=0)  # [D,]

    # backpropogation for dx
    dxhat = gamma * dgamma_xhat  # [N, D]
    dinvstd = np.sum(e * dxhat, axis=1).reshape((-1, 1))  # [N, 1]
    dstd = -1 / std**2 * dinvstd  # [N, 1]
    dvar = 0.5 / np.sqrt(var + eps) * dstd  # [N, 1]
    dse = 1/D * np.ones((N, D)) * dvar  # [N, D]
    de = 2 * e * dse  # [N, D]
    de += invstd * dxhat  # [N, D]
    dx = de  # [N, D]
    dmean = -1 * np.sum(de, axis=1).reshape((-1, 1))  # [N, 1]
    dx += 1/D * np.ones((N, D)) * dmean  # [N, D]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p  # mask of inverted dropout
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_conv = int(1 + (H + 2 * pad - HH) / stride)
    W_conv = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_conv, W_conv))

    # padding x
    x_pad = np.zeros((N, C, H + 2*pad, W + 2*pad))
    for i_N in range(N):
        for i_C in range(C):
            x_pad[i_N, i_C, :, :] = np.pad(x[i_N, i_C, :, :], pad, 'constant', constant_values=0)

    # forward pass of convolutional layer
    for i_N in range(N):
        for i_F in range(F):
            for i_H_conv in range(H_conv):
                for i_W_conv in range(W_conv):
                    H_start = i_H_conv * stride
                    H_end = i_H_conv * stride + HH
                    W_start = i_W_conv * stride
                    W_end = i_W_conv * stride + WW
                    x_patch = x_pad[i_N, :, H_start:H_end, W_start:W_end]
                    i_w = w[i_F, :, :, :]
                    out[i_N, i_F, i_H_conv, i_W_conv] = np.sum(x_patch * i_w) + b[i_F]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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

    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_conv = int(1 + (H + 2 * pad - HH) / stride)
    W_conv = int(1 + (W + 2 * pad - WW) / stride)

    # padding x
    x_pad = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for i_N in range(N):
        for i_C in range(C):
            x_pad[i_N, i_C, :, :] = np.pad(x[i_N, i_C, :, :], pad, 'constant', constant_values=0)

    # backprop
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for i_N in range(N):
        for i_F in range(F):
            for i_H_conv in range(H_conv):
                for i_W_conv in range(W_conv):
                    H_start = i_H_conv * stride
                    H_end = i_H_conv * stride + HH
                    W_start = i_W_conv * stride
                    W_end = i_W_conv * stride + WW
                    x_patch = x_pad[i_N, :, H_start:H_end, W_start:W_end]
                    i_w = w[i_F, :, :, :]
                    dw[i_F, :, :, :] += x_patch * dout[i_N, i_F, i_H_conv, i_W_conv]
                    dx_pad[i_N, :, H_start:H_end, W_start:W_end] += i_w * dout[i_N, i_F, i_H_conv, i_W_conv]
                    db[i_F] += dout[i_N, i_F, i_H_conv, i_W_conv]
    dx = dx_pad[:, :, 1:-1, 1:-1]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (W - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W= x.shape
    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    stride = pool_param['stride']
    H_maxpool = int(1 + (W - pool_H) / stride)
    W_maxpool = int(1 + (W - pool_W) / stride)

    # forward pass for maxpooling
    out = np.zeros((N, C, H_maxpool, W_maxpool))
    for i_N in range(N):
        for i_C in range(C):
            for i_H_max in range(H_maxpool):
                for i_W_max in range(W_maxpool):
                    H_start = i_H_max * stride
                    H_end = i_H_max * stride + pool_H
                    W_start = i_W_max * stride
                    W_end = i_W_max * stride + pool_W
                    x_patch = x[i_N, i_C, H_start:H_end, W_start:W_end]
                    out[i_N, i_C, i_H_max, i_W_max] = np.max(x_patch)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    _, _, H_maxpool, W_maxpool = dout.shape
    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    stride = pool_param['stride']

    # backprop for maxpooling
    dx = np.zeros_like(x)
    for i_N in range(N):
        for i_C in range(C):
            for i_H_max in range(H_maxpool):
                for i_W_max in range(W_maxpool):
                    H_start = i_H_max * stride
                    H_end = i_H_max * stride + pool_H
                    W_start = i_W_max * stride
                    W_end = i_W_max * stride + pool_W
                    x_patch = x[i_N, i_C, H_start:H_end, W_start:W_end]
                    dx_patch = np.zeros_like(x_patch).flatten()
                    dx_patch[np.argmax(x_patch)] = 1
                    dx_patch = dx_patch.reshape(x_patch.shape) * dout[i_N, i_C, i_H_max, i_W_max]
                    dx[i_N, i_C, H_start:H_end, W_start:W_end] += dx_patch

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_T = x.transpose(0, 2, 3, 1)
    x_T2d = np.reshape(x_T, (-1, x.shape[1]))
    out_T2d, cache = batchnorm_forward(x_T2d, gamma, beta, bn_param)
    out = np.reshape(out_T2d, x_T.shape).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

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

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dout_T = dout.transpose((0, 2, 3, 1))
    dout_T2d = np.reshape(dout_T, (-1, dout.shape[1]))
    dx_T2d, dgamma, dbeta = batchnorm_backward(dout_T2d, cache)
    dx = np.reshape(dx_T2d, dout_T.shape).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def groupnorm_forward(x, gamma, beta, G, gn_param):

    """
    Computes the forward pass for  group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner
    identical to that of batch normalization and layer normalization.

    Used in fully connected network

    Inputs:
    - x: Input data of shape (N, D)
    - gamma: Scale parameter, of shape (D,)
    - beta: Shift parameter, of shape (D,)
    - G: Integer number of groups to split into, should be a divisor of D
    - gn_param: Dictionary with the following keys:
    - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, D)
    - cache: Values needed for the backward pass
    """

    eps = gn_param.get('eps', 1e-5)

    N, D = x.shape
    x_g = np.reshape(x, (N, G, -1))

    # group normalization: forward pass
    mean_g = x_g.mean(axis=2)  # [N, G, 1]
    e_g = x_g - mean_g  # [N, G, D/G]
    var_g = np.mean(e_g ** 2, axis=2)  # [N, G, 1]
    std_g = np.sqrt(var_g + eps)  # [N, G, 1]
    invstd_g = 1 / std_g  # [N, G, 1]
    xhat_g = e_g * invstd_g  # [N, G, D/G]
    xhat = np.reshape(xhat_g, (N, -1))  # [N, D]
    out = gamma * xhat + beta  # [N, D]

    # cache
    cache = {'e_g': e_g, 'var_g': var_g, 'std_g': std_g, 'invstd_g': invstd_g,
             'xhat': xhat, 'gamma': gamma, 'D': D, 'N': N, 'G': G, 'eps': eps}

    return out, cache


def groupnorm_backward(dout, cache):

    """
    Computes the backward pass for group normalization.

    Used in fully connected network

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter, of shape (D,)
    - dbeta: Gradient with respect to shift parameter, of shape (D,)
    """

    # unpack cache
    e_g = cache['e_g']   # [N, G, D/G]
    var_g = cache['var_g']  # [N, G, 1]
    std_g = cache['std_g']  # [N, G, 1]
    invstd_g = cache['invstd_g']  # [N, G, 1]
    xhat = cache['xhat']  # [N, D]
    gamma = cache['gamma']  # [D,]
    N = cache['N']  # subjects
    D = cache['D']  # dimensions
    G = cache['G']  # groups, should be divisor of D
    eps = cache['eps']

    # backpropogation for dgamma and dbeta
    dbeta = np.sum(dout, axis=0)  # [D,]
    dgamma_xhat = dout  # [N, D]
    dgamma = np.sum(dgamma_xhat * xhat, axis=0)  # [D,]

    # backpropogation for dx
    dxhat = gamma * dgamma_xhat  # [N, D]
    dxhat_g = np.reshape(dxhat, (N, G, -1))  # [N, G, D/G]
    dinvstd_g = np.sum(e_g * dxhat_g, axis=2)  # [N, G, 1]
    dstd_g = -1 / std_g**2 * dinvstd_g  # [N, G, 1]
    dvar_g = 0.5 / np.sqrt(var_g + eps) * dstd_g  # [N, G, 1]
    dse_g = G/D * np.ones((N, G, D//G)) * dvar_g  # [N, G, D/G]
    de_g = 2 * e_g * dse_g  # [N, G, D/G]
    de_g += invstd_g * dxhat_g  # [N, G, D/G]
    dx_g = de_g  # [N, G, D/G]
    dmean_g = -1 * np.sum(de_g, axis=2)  # [N, G, 1]
    dx_g += G/D * np.ones((N, G, D//G)) * dmean_g  # [N, G, D/G]
    dx = np.reshape(dx_g, (N, -1))  # [N, D]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner
    identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer number of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """

    eps = gn_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x_g = np.reshape(x, (N, G, int(C/G), H, W))

    # spatial group normalization: forward pass
    mean_g = np.mean(x_g, axis=(2, 3, 4), keepdims=True)  # [N, G, 1, 1, 1]
    e_g = x_g - mean_g  # [N, G, C/G, H, W]
    var_g = np.mean(e_g ** 2, axis=(2, 3, 4), keepdims=True)  # [N, G, 1, 1 ,1]
    std_g = np.sqrt(var_g + eps)  # [N, G, 1, 1, 1]
    invstd_g = 1 / std_g  # [N, G, 1, 1, 1]
    xhat_g = e_g * invstd_g  # [N, G, C/G, H, W]
    xhat = np.reshape(xhat_g, (N, C, H, W))  # [N, C, H, W]
    out = gamma.reshape((1, C, 1, 1)) * xhat + beta.reshape((1, C, 1, 1))  # [N, C, H, W]

    # cache
    cache = {'e_g': e_g, 'var_g': var_g, 'std_g': std_g, 'invstd_g': invstd_g,
             'xhat': xhat, 'gamma': gamma, 'C': C, 'N': N, 'G': G,
             'H': H, 'W': W, 'eps': eps}

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # unpack cache
    e_g = cache['e_g']  # [N, G, C/G, H, W]
    var_g = cache['var_g']  # [N, G, 1, 1, 1]
    std_g = cache['std_g']   # [N, G, 1, 1, 1]
    invstd_g = cache['invstd_g']  # [N, G, 1, 1, 1]
    xhat = cache['xhat']  # [N, C, H, W]
    gamma = cache['gamma']  # [C,]
    N = cache['N']  # subjects
    C = cache['C']  # dimensions
    G = cache['G']  # groups, should be divisor of D
    H = cache['H']
    W = cache['W']
    eps = cache['eps']

    # backpropogation for dgamma and dbeta
    dbeta = np.sum(dout, axis=(0, 2, 3))  # [C,]
    dgamma_xhat = dout  # [N, C, H, W]
    dgamma = np.sum(dgamma_xhat * xhat, axis=(0, 2, 3))  # [C,]

    # backpropogation for dx
    dxhat = gamma.reshape((1, C, 1, 1)) * dgamma_xhat  # [N, C, H, W]
    dxhat_g = np.reshape(dxhat, (N, G, C//G, H, W))  # [N, G, C/G, H, W]
    dinvstd_g = np.sum(e_g * dxhat_g, axis=(2, 3, 4), keepdims=True)  # [N, G, 1, 1, 1]
    dstd_g = -1 / std_g**2 * dinvstd_g  # [N, G, 1, 1, 1]
    dvar_g = 0.5 / np.sqrt(var_g + eps) * dstd_g  # [N, G, 1, 1, 1]
    dse_g = (G / (C*H*W)) * np.ones((N, G, C//G, H, W)) * dvar_g  # [N, G, C/G, H, W]
    de_g = 2 * e_g * dse_g  # [N, G, C/G, H, W]
    de_g += invstd_g * dxhat_g  # [N, G, C/G, H, W]
    dx_g = de_g  # [N, G, C/G, H, W]
    dmean_g = -1 * np.sum(de_g, axis=(2, 3, 4), keepdims=True)  # [N, G, 1, 1, 1]
    dx_g += (G / (C*H*W)) * np.ones((N, G, C//G, H, W)) * dmean_g  # [N, G, C/G, H, W]
    dx = np.reshape(dx_g, (N, C, H, W))  # [N, C, H, W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]

    # only the log_probs of corrected class go into loss
    # SGD is fulfilled by deviding N
    loss = -np.sum(log_probs[np.arange(N), y]) / N

    # derivative
    dx = probs.copy()
    dx[np.arange(N), y] -= 1  # derivative of cross-entropy loss (softmax loss)
    dx /= N  # SGD (mini-batch GD)
    return loss, dx
