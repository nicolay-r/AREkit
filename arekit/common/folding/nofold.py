from arekit.common.folding.base import BaseDataFolding


class NoFolding(BaseDataFolding):
    """ The case of absent folding in experiment.
    """

    def __init__(self, doc_ids_to_fold, supported_data_types, dup_count=1):
        assert(isinstance(dup_count, int) and dup_count > 0)

        if len(supported_data_types) > 1:
            raise NotImplementedError("Experiments with such amount of data-types are not supported!")

        super(NoFolding, self).__init__(doc_ids_to_fold=doc_ids_to_fold,
                                        supported_data_types=supported_data_types,
                                        states_count=dup_count)

    @property
    def Name(self):
        return "na"

    def fold_doc_ids_set(self):
        return {
            self._supported_data_types[0]: list(self._doc_ids_to_fold_set)
        }
