from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding


class SingleDocOperations(DocumentOperations):
    """ Operations over a single document for inference.
    """

    def iter_tagget_doc_ids(self, tag):
        assert(isinstance(tag, BaseDocumentTag))
        assert(tag == BaseDocumentTag.Annotate)
        return [0]

    def __init__(self, news, text_parser):
        folding = NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test])
        super(SingleDocOperations, self).__init__(folding, text_parser)
        self.__doc = news

    def get_doc(self, doc_id):
        return self.__doc