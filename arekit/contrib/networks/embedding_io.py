class BaseEmbeddingIO(object):
    """ API for loading and saving embedding and vocabulary related data.
    """

    def save_vocab(self, data, data_folding):
        raise NotImplementedError()

    def load_vocab(self, data_folding):
        raise NotImplementedError()

    def save_embedding(self, data, data_folding):
        raise NotImplementedError()

    def load_embedding(self, data_folding):
        raise NotImplementedError()

    def check_targets_existed(self, data_folding):
        raise NotImplementedError()
