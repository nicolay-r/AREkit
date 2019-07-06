from base import CommonModelSettings


class CellTypes:
    RNN = u'vanilla'
    LSTM = u'lstm'
    GRU = u'gru'


class RNNConfig(CommonModelSettings):

    __hidden_size = 300
    __cell_type = CellTypes.LSTM
    __l2_reg_lambda = 0.0

    @property
    def L2RegLambda(self):
        return self.__l2_reg_lambda

    @property
    def CellType(self):
        return self.__cell_type

    @property
    def HiddenSize(self):
        return self.__hidden_size

    @property
    def LearningRate(self):
        return 0.01

    def set_cell_type(self, cell_type):
        assert(isinstance(cell_type, unicode))
        self.__cell_type = cell_type

    def modify_hidden_size(self, value):
        assert(isinstance(value, int) and value > 0)
        self.__hidden_size = value

    def _internal_get_parameters(self):
        parameters = super(RNNConfig, self)._internal_get_parameters()

        parameters += [
            ("rnn:hidden_size", self.HiddenSize),
            ("rnn:cell_type", self.CellType),
            ("rnn:l2_reg_lambda", self.L2RegLambda)
        ]

        return parameters
