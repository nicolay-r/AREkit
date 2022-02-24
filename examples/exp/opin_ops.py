from arekit.common.experiment.api.ops_opin import OpinionOperations
from arekit.common.opinions.collection import OpinionCollection


class CustomOpinionOperations(OpinionOperations):

    def __init__(self, labels_formatter, exp_io, synonyms, neutral_labels_fmt):
        super(CustomOpinionOperations, self).__init__()
        self.__labels_formatter = labels_formatter
        self.__exp_io = exp_io
        self.__synonyms = synonyms
        self.__neutral_labels_fmt = neutral_labels_fmt

    @property
    def LabelsFormatter(self):
        return self.__labels_formatter

    def iter_opinions_for_extraction(self, doc_id, data_type):
        # Reading automatically annotated collection of neutral opinions.
        # TODO. #250, #251 provide opinion annotation here for the particular document.
        return self.__exp_io.read_opinion_collection(
            target=self.__exp_io.create_result_opinion_collection_target(
                doc_id=doc_id,
                data_type=data_type,
                epoch_index=0),
            labels_formatter=self.__neutral_labels_fmt,
            create_collection_func=self.create_opinion_collection)

    def get_etalon_opinion_collection(self, doc_id):
        return self.create_opinion_collection(None)

    def get_result_opinion_collection(self, doc_id, data_type, epoch_index):
        raise Exception("Not Supported")

    def create_opinion_collection(self, opinions=None):
        return OpinionCollection(opinions=[] if opinions is None else opinions,
                                 synonyms=self.__synonyms,
                                 error_on_duplicates=True,
                                 error_on_synonym_end_missed=True)
