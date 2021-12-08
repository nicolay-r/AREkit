class Stemmer:
    """
    Interface
    """

    def lemmatize_to_list(self, text):
        raise NotImplementedError()

    def lemmatize_to_str(self, text):
        raise NotImplementedError()

    def is_adjective(self, pos_type):
        raise NotImplementedError()

    def is_noun(self, pos_type):
        raise NotImplementedError()
