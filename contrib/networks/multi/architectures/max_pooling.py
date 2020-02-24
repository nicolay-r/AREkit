import tensorflow as tf
from collections import OrderedDict
from arekit.networks.multi.architectures.base_single_mlp import BaseMultiInstanceSingleMLP


class MaxPoolingOverSentences(BaseMultiInstanceSingleMLP):
    """
    Title: Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks
    Authors: Xiaotian Jiang, Quan Wang, Peng Li, Bin Wang
    Paper: https://pdfs.semanticscholar.org/8731/369a707046f3f8dd463d1fd107de31d40a24.pdf
    """

    def __init__(self, context_network):
        super(MaxPoolingOverSentences, self).__init__(context_network)
        self.__hidden = OrderedDict()

    @property
    def EmbeddingSize(self):
        return self.ContextNetwork.ContextEmbeddingSize

    def init_multiinstance_embedding(self, context_outputs):
        """
        context_outputs: [batches, sentences, embedding]
        """
        return self.__contexts_max_pooling(context_outputs=context_outputs,
                                           contexts_per_opinion=self.ContextsPerOpinion)  # [batches, max_pooling]

    @staticmethod
    def __contexts_max_pooling(context_outputs, contexts_per_opinion):
        context_outputs = tf.expand_dims(context_outputs, 0)     # [1, batches, sentences, embedding]
        context_outputs = tf.nn.max_pool(
            context_outputs,
            ksize=[1, 1, contexts_per_opinion, 1],
            strides=[1, 1, contexts_per_opinion, 1],
            padding='VALID',
            data_format="NHWC")
        return tf.squeeze(context_outputs)                       # [batches, max_pooling]
