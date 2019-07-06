class NetworkIO(object):
    """
    Now it includes IO to interact with collection,
    and it is specific towards RuSentiRel collection.
    """

    def get_etalon_root(self):
        raise Exception("Not implemented")

    def get_word_embedding_filepath(self):
        raise Exception("Not implemented")

    def get_rusentiframes_collection_filepath(self):
        raise Exception("Not implemented")

    def read_opinion_collection_by_filepath(self, filepath, synonyms):
        raise Exception("Not implemented")

    def iter_opinion_collections_to_compare(self, data_type, etalon_root, indices, synonyms):
        raise Exception("Not implemented")

    def get_opinion_output_filepath(self, data_type, article_index):
        raise Exception("Not implemented")

    def get_model_state_filepath(self):
        raise Exception("Not implemented")
