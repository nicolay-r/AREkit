import tensorflow as tf


class CellTypes:
    RNN = u'vanilla'
    GRU = u'gru'
    LSTM = u'lstm'
    BasicLSTM = u'basic-lstm'


def get_cell(hidden_size, cell_type, lstm_initializer=None, dropout_rnn_keep_prob=1.0):
    assert(isinstance(cell_type, unicode))

    cell = None

    if cell_type == CellTypes.RNN:
        cell = tf.nn.rnn_cell.BasicRNNCell(
            num_units=hidden_size)

    elif cell_type == CellTypes.LSTM:
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units=hidden_size,
            initializer=lstm_initializer)

    elif cell_type == CellTypes.BasicLSTM:
        cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units=hidden_size)

    elif cell_type == CellTypes.GRU:
        cell = tf.nn.rnn_cell.GRUCell(
            num_units=hidden_size)
    else:
        Exception("Incorrect cell_type={}".format(cell_type))

    dropped_cell = tf.nn.rnn_cell.DropoutWrapper(
        cell=cell,
        output_keep_prob=dropout_rnn_keep_prob)

    return dropped_cell


def rnn(cell, inputs, sequence_length, dtype, scope=None):
    """
    Difference between static and dynamic rnn's
    https://stackoverflow.com/questions/51425803/what-is-the-difference-between-static-rnn-and-dynamic-rnn
    """
    return tf.nn.dynamic_rnn(cell=cell,
                             inputs=inputs,
                             sequence_length=sequence_length,
                             dtype=dtype,
                             scope=scope)


def bidirectional_rnn(cell_fw, cell_bw, inputs, sequence_length, dtype):
    """
    Difference between static and dynamic rnn's
    https://stackoverflow.com/questions/51425803/what-is-the-difference-between-static-rnn-and-dynamic-rnn
    """
    return tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                           cell_bw=cell_bw,
                                           inputs=inputs,
                                           sequence_length=sequence_length,
                                           dtype=dtype)


def select_last_relevant_in_sequence(sequence, length):
    assert(isinstance(sequence, tf.Tensor))
    assert(isinstance(length, tf.Tensor))

    batch_size = tf.shape(sequence)[0]
    max_length = int(sequence.get_shape()[1])
    input_size = int(sequence.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(sequence, [-1, input_size])
    return tf.gather(flat, index)


def calculate_sequence_length(sequence, is_neg_placeholder=False):
    """
    By default, considering that '0' (zeros) are placeholder in sequences.

    Considering that '-1' or negative values are
    placeholder in sequences.
    """
    assert(isinstance(sequence, tf.Tensor))

    relevant = tf.sign(tf.abs(sequence))

    if is_neg_placeholder:
        relevant = tf.add(relevant, tf.constant(value=1, shape=sequence.shape))

    length = tf.reduce_sum(relevant, reduction_indices=1)
    length = tf.cast(length, tf.int32)

    return length
