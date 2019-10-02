import tensorflow as tf
from core.networks.context.configurations.rnn import CellTypes


def get_cell(hidden_size, cell_type, lstm_initializer=None):
    assert(isinstance(cell_type, unicode))

    if cell_type == CellTypes.RNN:
        return tf.nn.rnn_cell.BasicRNNCell(
            num_units=hidden_size)

    elif cell_type == CellTypes.LSTM:
        return tf.nn.rnn_cell.LSTMCell(
            num_units=hidden_size,
            initializer=lstm_initializer)

    elif cell_type == CellTypes.BasicLSTM:
        return tf.nn.rnn_cell.BasicLSTMCell(
            num_units=hidden_size)

    elif cell_type == CellTypes.GRU:
        return tf.nn.rnn_cell.GRUCell(
            num_units=hidden_size)

    Exception("Incorrect cell_type={}".format(cell_type))
    return None
