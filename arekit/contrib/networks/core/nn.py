class NeuralNetwork(object):

    @property
    def ParametersDictionary(self):
        return {}

    @property
    def Cost(self):
        raise NotImplementedError()

    @property
    def Labels(self):
        raise NotImplementedError()

    @property
    def Accuracy(self):
        raise NotImplementedError()

    def iter_hidden_parameters(self):
        raise NotImplementedError()

    def iter_input_dependent_hidden_parameters(self):
        return
        yield

    def compile(self, config, reset_graph, graph_seed):
        raise NotImplementedError()

    def create_feed_dict(self, input, data_type):
        raise NotImplementedError()
