from arekit.common.experiment.cv.base import BaseCVFolding
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

        data_types_splits = self.__folding_algo.get_cv_split(
            doc_ids_iter=data_indices,
            data_types=list(self.iter_supported_data_types()))

        for doc_id in data_types_splits[data_type]:
            yield doc_id
