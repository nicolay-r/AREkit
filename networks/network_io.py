class NetworkIO(object):
    """
    Now it includes IO to interact with collection,
    and it is specific towards RuSentiRel collection.
    """

    def get_etalon_root(self):
        raise NotImplementedError()

    def get_word_embedding_filepath(self):
        raise NotImplementedError()

    def get_rusentiframes_collection_filepath(self):
        raise NotImplementedError()

    def read_opinion_collection_by_filepath(self, filepath, synonyms):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, etalon_root, indices, synonyms):
        raise NotImplementedError()

    def get_opinion_output_filepath(self, data_type, article_index):
        raise NotImplementedError()

    def get_model_state_filepath(self):
        raise NotImplementedError()

