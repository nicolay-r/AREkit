from arekit.common.folding.base import BaseDataFolding


class NoFolding(BaseDataFolding):
    """ The case of absent folding in experiment.
    """

    def __init__(self, doc_ids, supported_data_type):
        super(NoFolding, self).__init__(doc_ids_to_fold=doc_ids,
                                        supported_data_types=[supported_data_type])

    def fold_doc_ids_set(self):
        return {
            self._supported_data_types[0]: list(self._doc_ids_to_fold_set)
        }
