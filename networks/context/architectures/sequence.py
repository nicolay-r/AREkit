import tensorflow as tf
from core.networks.context.configurations.rnn import CellTypes


def get_cell(hidden_size, cell_type, lstm_initializer=None, dropout_rnn_keep_prob=1.0):
    assert(isinstance(cell_type, unicode))
    assert(isinstance(dropout_rnn_keep_prob, float))

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
