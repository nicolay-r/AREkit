from arekit.common.experiment.data_type import DataType
from arekit.common.folding.base import BaseDataFolding


class NoFolding(BaseDataFolding):
    """ The case of absent folding in experiment.
    """

    def fold_doc_ids_set(self, doc_ids):
        assert(isinstance(doc_ids, dict) and len(doc_ids) == 1)
        assert(isinstance(list(doc_ids.keys())[0], DataType))
        return doc_ids
