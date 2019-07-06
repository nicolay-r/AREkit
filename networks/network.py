class NeuralNetwork(object):

    @property
    def ParametersDictionary(self):
        """
        returns: dict
            Dictionary that illustrates a parameter values by it's string keys
        """
        return {}

    @property
    def Cost(self):
        """
        returns: Tensor
            Error result
        """
        raise Exception("Not implemented")

    @property
    def Labels(self):
        """
        returns: Tensor
            Result labels by passed batch through the neural network
        """
        raise Exception("Not implemented")

    # TODO. rename like: get_variables_to_investigate
    @property
    def Variables(self):
        """
        return: list, list
            parameter names and perameter values
        """
        return [], []

    @property
    def Accuracy(self):
        raise Exception("Not implemented")

    def compile(self, config, reset_graph):
        """
        compile tf graph
        """
        raise Exception("Not implemented")

    def create_feed_dict(self, input, data_type):
        raise Exception("Not implemented")
