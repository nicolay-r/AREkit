class NetworkIO(object):
    """
    Now it includes IO to interact with collection,
    and it is specific towards RuSentiRel collection.
    """

    @property
    def SynonymsCollection(self):
        raise NotImplementedError()

    def get_model_root(self):
        """
        Considering a root with all the results and hidden states.
        """
        raise NotImplementedError()

    def get_model_filepath(self):
        raise NotImplementedError()

    def get_word_embedding_filepath(self):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        raise NotImplementedError()

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        raise NotImplementedError()

    def create_model_state_filepath(self):
        raise NotImplementedError()

    def write_log(self, log_names, log_values):
        raise NotImplementedError()

