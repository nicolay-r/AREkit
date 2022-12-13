class BaseVectorizer(object):
    """ Custom API for vectorization
    """

    def create_term_embedding(self, term):
        raise NotImplementedError()
