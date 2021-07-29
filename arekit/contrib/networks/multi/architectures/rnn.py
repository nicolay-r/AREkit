from arekit.contrib.networks.multi.architectures.base.base import BaseMultiInstanceNeuralNetwork


class RNNOverSentences(BaseMultiInstanceNeuralNetwork):
    """
    TODO. Implement the same way as rnn in ctx.
    """

    @property
    def EmbeddingSize(self):
        return self.ContextNetwork.ContextEmbeddingSize

    pass
