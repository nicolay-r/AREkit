import collections

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding


class NoFolding(BaseDataFolding):
    """ The case of absent folding in experiment.
    """

    def __init__(self, data_type):
        assert(isinstance(data_type, DataType))
        super(NoFolding, self).__init__(supported_data_types=[data_type])
        self.__data_type = data_type

    def fold_doc_ids_set(self, doc_ids):
        assert(isinstance(doc_ids, collections.Iterable))
        return {
            self.__data_type: list(set(doc_ids))
        }
