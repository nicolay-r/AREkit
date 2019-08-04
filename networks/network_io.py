class NetworkIO(object):
    """
    Now it includes IO to interact with collection,
    and it is specific towards RuSentiRel collection.
    """

    def get_word_embedding_filepath(self):
        raise NotImplementedError()

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, synonyms):
        raise NotImplementedError()

    def create_result_opinion_collection_filepath(self, data_type, doc_id):
        raise NotImplementedError()

    def create_model_state_filepath(self):
        raise NotImplementedError()

    def get_answer_filepath(self, doc_id):
        raise NotImplementedError()

    def get_hidden_parameter_filepath(self, parameter_name, epoch):
        raise NotImplementedError()
