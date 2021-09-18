from arekit.common.experiment.folding.base import BaseExperimentDataFolding


class FixedFolding(BaseExperimentDataFolding):

    def __init__(self, doc_to_dtype_func, doc_ids_to_fold, supported_data_types):
        assert(callable(doc_to_dtype_func))
        super(FixedFolding, self).__init__(doc_ids_to_fold=doc_ids_to_fold,
                                           supported_data_types=supported_data_types)

        self.__doc_to_dtype_func = doc_to_dtype_func

    @property
    def Name(self):
        return "fixed"

    def fold_doc_ids_set(self):

        folded = {}
        for d_type in self._supported_data_types:
            folded[d_type] = []

        for doc_id in self._doc_ids_to_fold_set:
            d_type = self.__doc_to_dtype_func(doc_id)
            folded[d_type].append(doc_id)

        return folded

    def get_current_state(self):
        """ Returns in order to be compatible with cv-based experiment format.
        """
        return "0"
