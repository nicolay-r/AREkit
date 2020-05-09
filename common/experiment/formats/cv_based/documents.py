from arekit.common.experiment.cv.base import BaseCVFolding
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations


class CVBasedDocumentOperations(DocumentOperations):
    """
    Limitations: provides separation onto Train/Test collections.
    """

    def __init__(self, folding_algo):
        assert(isinstance(folding_algo, BaseCVFolding))
        self.__folding_algo = folding_algo

    @property
    def _FoldingAlgo(self):
        return self.__folding_algo

    def get_data_indices_to_fold(self):
        raise NotImplementedError()

    def iter_news_indices(self, data_type):
        data_indices = self.get_data_indices_to_fold()
        train, test = self.__folding_algo.get_cv_train_test_pair_by_index(doc_ids_iter=data_indices)

        if data_type not in [DataType.Train, DataType.Test]:
            raise Exception("Not supported data_type='{data_type}'".format(data_type=data_type))

        result_list = train if data_type == DataType.Train else test

        for doc_id in result_list:
            yield doc_id

