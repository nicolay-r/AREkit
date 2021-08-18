from arekit.common.experiment.formats.opinions import OpinionOperations


class DefaultOpinionOperations(OpinionOperations):

    def try_read_annotated_opinion_collection(self, doc_id, data_type):
        pass

    def save_annotated_opinion_collection(self, collection, doc_id, data_type):
        pass

    def iter_opinions_for_extraction(self, doc_id, data_type):
        pass

    def read_etalon_opinion_collection(self, doc_id):
        pass

    def read_result_opinion_collection(self, data_type, doc_id, epoch_index):
        pass

    def create_opinion_collection(self):
        pass