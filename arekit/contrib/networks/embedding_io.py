class BaseEmbeddingIO(object):
    """ API for loading and saving embedding and vocabulary related data.
    """

    def save_vocab(self, data):
        raise NotImplementedError()

    def load_vocab(self,):
        raise NotImplementedError()

    def save_embedding(self, data):
        raise NotImplementedError()

    def load_embedding(self):
        raise NotImplementedError()

    def check_targets_existed(self):
        raise NotImplementedError()
