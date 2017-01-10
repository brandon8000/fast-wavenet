import numpy as np
import tensorflow as tf

def time_to_batch(inputs, rate, pad = True):
    """If necessary zero-pads inputs and reshape by rate.

    Used to perform 1D dilated convolution.

    Args:
      inputs: (tensor)
      rate: (int)
    Outputs:
      outputs: (tensor)
      pad_left: (int)
    """
    _, width, num_channels = inputs.get_shape().as_list()
    if pad:
        width_pad = int(rate * np.ceil((width + rate) * 1.0 / rate))
        pad_left = width_pad - width
        padded = tf.pad(inputs, [[0, 0], [pad_left, 0], [0, 0]])
    else:
        width_pad = width
        padded = inputs
    perm = (1, 0, 2)
    shape = (width_pad // rate, -1, num_channels)
    transposed = tf.transpose(padded, perm)
    reshaped = tf.reshape(transposed, shape)
    outputs = tf.transpose(reshaped, perm)
    return outputs


def batch_to_time(inputs, rate, crop_left = 0):
    """ Reshape to 1d signal, and remove excess zero-padding.

    Used to perform 1D dilated convolution.

    Args:
      inputs: (tensor)
      crop_left: (int)
      rate: (int)
    Ouputs:
      outputs: (tensor)
    """
    shape = tf.shape(inputs)
    batch_size = shape[0] / rate
    width = shape[1]
    out_width = tf.to_int32(width * rate)
    _, _, num_channels = inputs.get_shape().as_list()
    perm = (1, 0, 2)
    new_shape = (out_width, -1, num_channels)
    transposed = tf.transpose(inputs, perm)
    reshaped = tf.reshape(transposed, new_shape)
    outputs = tf.transpose(reshaped, perm)
    cropped = tf.slice(outputs, [0, crop_left, 0], [-1, -1, -1])
    return cropped


def conv1d(inputs, out_channels, 
           filter_width = 2, 
           stride = 1, 
           padding = 'VALID',
           data_format = 'NHWC',
           gain = np.sqrt(2), 
           activation = tf.nn.relu, 
           bias = False,
           name = ''):
    """One dimension convolution helper function.

    Sets variables with good defaults.

    Args:
      inputs:
      out_channels:
      filter_width:
      stride:
      paddding:
      data_format:
      gain:
      activation:
      bias:

    Outputs:
      outputs:
    """
    in_channels = inputs.get_shape().as_list()[-1]
    stddev = gain / np.sqrt(filter_width ** 2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev)
    ww = tf.get_variable(name='w_' + name, shape=(filter_width, in_channels, out_channels), initializer=w_init)
    outputs = tf.nn.conv1d(inputs, ww, stride=stride, padding=padding, data_format=data_format)
    if bias:
        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable(name='b_' + name, shape=(out_channels,), initializer=b_init)
        outputs = outputs + tf.expand_dims(tf.expand_dims(b, 0), 0)
    if activation:
        outputs = activation(outputs)
    return outputs


def conv1d_with_weight(inputs,
                       out_channels,
                       filter_width = 2, 
                       stride = 1, 
                       padding = 'VALID', 
                       data_format = 'NHWC',
                       gain = np.sqrt(2), 
                       activation = tf.nn.relu,bias = False, 
                       name = ''):
    """One dimension convolution helper function.

    Sets variables with good defaults.

    Args:
      inputs:
      out_channels:
      filter_width:
      stride:
      paddding:
      data_format:
      gain:
      activation:
      bias:

    Outputs:
      outputs:
    """
    in_channels = inputs.get_shape().as_list()[-1]
    stddev = gain / np.sqrt(filter_width ** 2 * in_channels)
    w_init = tf.random_normal_initializer(stddev=stddev)
    ww = tf.get_variable(name='w_' + name, shape=(filter_width, in_channels, out_channels), initializer=w_init)
    outputs = tf.nn.conv1d(inputs, ww, stride=stride, padding=padding, data_format=data_format)
    if bias:
        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable(name='b_' + name, shape=(out_channels,), initializer=b_init)
        outputs = outputs + tf.expand_dims(tf.expand_dims(b, 0), 0)
    if activation:
        outputs = activation(outputs)
    return (outputs, ww)


def dilated_conv1d(inputs,
                   gated_channels,
                   out_channels, 
                   skip_channels,
                   filter_width = 2, 
                   rate = 1, 
                   padding = 'VALID', 
                   name = None, 
                   gain = np.sqrt(2)):
    """

    Args:
      inputs: (tensor)
      output_channels: num channels at the gated output
      filter_width:
      rate:
      padding:
      name:
      gain:

    Outputs:
      outputs: (tensor)

    Implements the following architecture
       input (size = in_channel) -> 1x1 conv (optional) -> input_proc(size=out_channel) -...

                               |-> [gate]   -|                              |-> 1x1 conv -> skip output (size = skip_channel)
                               |             |-> (*)(size = gated_channel) -|
                              -|-> [filter] -|                              |-> 1x1 conv -|
                               |                                                          |-> (+) -> dense output (size = out_channel)
                               |----------------------------------------------------------|
    """
    _, width, in_channels = inputs.get_shape().as_list()
    assert name
    with tf.variable_scope(name):
        if in_channels != out_channels:
            inputs_proc = conv1d(inputs, 
                                 out_channels=out_channels, 
                                 filter_width=1, 
                                 padding=padding, 
                                 gain=gain, 
                                 activation=None,
                                 name='pre')
        else:
            inputs_proc = inputs
            
        inputs_ = time_to_batch(inputs_proc, rate=rate)
        outputs_filter = conv1d(inputs_, 
                                out_channels=gated_channels, 
                                filter_width=filter_width, 
                                padding=padding, 
                                gain=gain,
                                activation=tf.tanh, 
                                name='filter')
        outputs_gate = conv1d(inputs_, 
                              out_channels=gated_channels, 
                              filter_width=filter_width, 
                              padding=padding, 
                              gain=gain,
                              activation=tf.sigmoid, 
                              name='gate')
        outputs_gated = tf.mul(outputs_filter, outputs_gate)
        
        
        _, conv_out_width, _ = outputs_gated.get_shape().as_list()
        new_width = conv_out_width * rate
        diff = new_width - width
        outputs = batch_to_time(outputs_gated, rate=rate, crop_left=diff)
        
        outputs_residual = conv1d(outputs, 
                                  out_channels=out_channels, 
                                  filter_width=1, 
                                  padding=padding, 
                                  gain=1, 
                                  activation=None, 
                                  name='residual')
        outputs_dense = tf.add(inputs_proc, outputs_residual)
        
        outputs_skip = conv1d(outputs, 
                              out_channels=skip_channels, 
                              filter_width=1, padding=padding, 
                              gain=1, activation=None, name='skip')
        
        
        tensor_shape_dense = [tf.Dimension(None), tf.Dimension(width), tf.Dimension(out_channels)]
        outputs_dense.set_shape(tf.TensorShape(tensor_shape_dense))
        tensor_shape_skip = [tf.Dimension(None), tf.Dimension(width), tf.Dimension(skip_channels)]
        outputs_skip.set_shape(tf.TensorShape(tensor_shape_skip))
    return (outputs_dense, outputs_skip, outputs_gated)


def dilated_conv1d_nopad(inputs, 
                         gated_channels,
                         out_channels, 
                         skip_channels, 
                         skip_width, 
                         filter_width = 2, 
                         rate = 1, 
                         padding = 'VALID', 
                         name = None, 
                         gain = np.sqrt(2)):
    """

    Args:
      inputs: (tensor)
      output_channels: num channels at the gated output
      filter_width:
      rate:
      padding:
      name:
      gain:

    Outputs:
      outputs: (tensor)

    Implements the following architecture
       input (size = in_channel) -> 1x1 conv (optional) -> input_proc(size=out_channel) -...

                               |-> [gate]   -|                              |-> 1x1 conv -> skip output (size = skip_channel)
                               |             |-> (*)(size = gated_channel) -|
                              -|-> [filter] -|                              |-> 1x1 conv -|                                                                                |                                                          |-> (+) -> dense output (size = out_channel)
                               |----------------------------------------------------------|
    """
    _, width, in_channels = inputs.get_shape().as_list()
    assert width % rate == 0, 'input width should be multiples of rate'
    assert name
    with tf.variable_scope(name):
        if in_channels != out_channels:
            inputs_proc = conv1d(inputs, 
                                 out_channels=out_channels, 
                                 filter_width =1, 
                                 padding=padding, 
                                 gain=gain, 
                                 activation=None, 
                                 name='pre')
        else:
            inputs_proc = inputs
            
        inputs_ = time_to_batch(inputs_proc, rate=rate, pad=False)
        outputs_filter = conv1d(inputs_, 
                                out_channels=gated_channels,
                                filter_width=filter_width, 
                                padding=padding, 
                                gain=gain,
                                activation=tf.tanh, 
                                name='filter')
        outputs_gate = conv1d(inputs_, 
                              out_channels=gated_channels, 
                              filter_width=filter_width, 
                              padding=padding, 
                              gain=gain, 
                              activation=tf.sigmoid,
                              name='gate')
        outputs_gated = tf.mul(outputs_filter, outputs_gate)
        
        outputs = batch_to_time(outputs_gated, rate=rate)
        
        outputs_residual = conv1d(outputs, 
                                  out_channels=out_channels, 
                                  filter_width=1, 
                                  padding=padding,
                                  gain=1, 
                                  activation=None, 
                                  name='residual')
        outputs_dense = tf.add(inputs_proc[:, rate:, :], outputs_residual)
        
        outputs_skip = conv1d(outputs[:, -skip_width:, :],
                              out_channels=skip_channels,
                              filter_width=1, 
                              padding=padding, 
                              gain=1, 
                              activation=None, 
                              name='skip')
        
        tensor_shape_dense = [tf.Dimension(None), tf.Dimension(width - rate), tf.Dimension(out_channels)]
        outputs_dense.set_shape(tf.TensorShape(tensor_shape_dense))
        tensor_shape_skip = [tf.Dimension(None), tf.Dimension(skip_width), tf.Dimension(skip_channels)]
        outputs_skip.set_shape(tf.TensorShape(tensor_shape_skip))
    return (outputs_dense, outputs_skip, inputs_proc)


def post_processing(inputs, num_layers, num_classes):
    """ Performs post-processing (fully connected layers, 1 X 1 convolutions)
    inputs: a list of skip outputs of each dialted layer
    num_layers: number of layers, including the final output
    num_classes: the dimension of the final output """
    in_channels = inputs[0].get_shape().as_list()[-1]
    inputs_agg = sum(inputs)
    h = tf.nn.relu(inputs_agg)
    
    for l in range(num_layers - 1):
        with tf.variable_scope('post_l{}'.format(l)):
            h = conv1d(h, 
                       out_channels=in_channels, 
                       filter_width=1, 
                       padding='VALID', 
                       gain=1, 
                       activation=tf.nn.relu)

    with tf.variable_scope('post_l{}'.format(num_layers - 1)):
        outputs = conv1d(h,
                         out_channels=num_classes, 
                         filter_width=1, 
                         padding='VALID', 
                         gain=1, 
                         activation=None)
    return outputs


def _causal_linear(inputs, 
                   state, 
                   order, 
                   bias = False, 
                   name = '', 
                   activation = None):
    """Performs efficient causal filter generation

    Inputs:
    intputs: the inputs tensor of size (batch_size, input_channels) at the specific time stamp
    state: the recurrent state tensor of size (batch_size, state_channels)
    bias: true if bias is included
    order: the order of the convolution filters, currently supporting only 1 or 2
    name: the name of the suffix of the variable
    activation: the activation function applied

    Returns:
    output: the output of the convolution at that particular time
    """
    
    w = tf.get_variable('w_' + name)
    if order == 2:
        w_r = w[0, :, :]
        w_e = w[1, :, :]
        output = tf.matmul(inputs, w_e) + tf.matmul(state, w_r)
    else:
        w = w[0, :, :]
        output = tf.matmul(inputs, w)
    if bias:
        b = tf.get_variable('b_' + name)
        output = tf.add(output_filter, b)
    if activation:
        output = activation(output)
    return output


def dilated_generation(inputs, state, name = None):
    """
    Implements the following architecture, for efficient realtime generation

                               |-> [gate]   -|                              |-> 1x1 conv -> skip output (size = skip_channel)
                               |             |-> (*)(size = gated_channel) -|
    input (size = in_channel) -|-> [filter] -|                              |-> 1x1 conv -|
                               |                                                          |-> (+) -> dense output (size = out_channel)
                               |----------------------------------------------------------|
    """
    assert name
    with tf.variable_scope(name, reuse=True) as scope:
        try:
            inputs_ = _causal_linear(inputs, None, 1, name='pre', activation=None)
            state_ = _causal_linear(state, None, 1, name='pre', activation=None)
        except ValueError:
            inputs_ = inputs
            state_ = state

        output_filter = _causal_linear(inputs_, state_, 2, name='filter', activation=tf.tanh)
        output_gate = _causal_linear(inputs_, state_, 2, name='gate', activation=tf.sigmoid)
        output_gated = tf.mul(output_filter, output_gate)
        output_residual = _causal_linear(output_gated, None, 1, name='residual',activation=None)
        output_dense = tf.add(inputs_, output_residual)
        output_skip = _causal_linear(output_gated, None, 1, name='skip', activation=None)
    return (output_dense, output_skip, output_gated)


def post_processing_generation(inputs, num_layers, num_classes):
    """ Performs post-processing (fully connected layers, 1 X 1 convolutions) for efficient generation
    inputs: a list of skip outputs of each dialted layer
    num_layers: number of layers, including the final output
    num_classes: the dimension of the final output """
    
    in_channels = inputs[0].get_shape().as_list()[-1]
    inputs_agg = sum(inputs)
    h = tf.nn.relu(inputs_agg)
    for l in range(num_layers - 1):
        with tf.variable_scope('post_l{}'.format(l), reuse=True):
            h = _causal_linear(h, None, 1, activation=tf.nn.relu)

    with tf.variable_scope('post_l{}'.format(num_layers - 1), reuse=True):
        outputs = _causal_linear(h, None, 1)
    return outputs


def _output_linear(h, name = ''):
    with tf.variable_scope(name, reuse=True):
        w = tf.get_variable('w')[0, :, :]
        b = tf.get_variable('b')
        output = tf.matmul(h, w) + tf.expand_dims(b, 0)
    return output
