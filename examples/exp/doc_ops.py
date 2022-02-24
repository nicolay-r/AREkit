from arekit.common.experiment.api.enums import BaseDocumentTag
from arekit.common.experiment.api.ops_doc import DocumentOperations


class SingleDocOperations(DocumentOperations):
    """ Operations over a single document for inference.
    """

    def iter_tagget_doc_ids(self, tag):
        assert(isinstance(tag, BaseDocumentTag))
        assert(tag == BaseDocumentTag.Annotate)
        return [0]

    def __init__(self, news, exp_ctx, text_parser):
        super(SingleDocOperations, self).__init__(exp_ctx, text_parser)
        self.__doc = news

    def get_doc(self, doc_id):
        return self.__doc