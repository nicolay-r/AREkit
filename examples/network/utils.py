from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding


class SingleDocOperations(DocumentOperations):
    """ Operations over a single document for inference.
    """

    # TODO. 212. Rename, add tag.
    def iter_doc_ids_to_annotate(self):
        return 0

    # TODO. 212. Remove (we don't need it in such case).
    def iter_doc_ids_to_compare(self):
        raise NotImplementedError()

    def __init__(self, news):
        folding = NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test])
        super(SingleDocOperations, self).__init__(folding)
        self.__doc = news

    def get_doc(self, doc_id):
        return self.__doc
