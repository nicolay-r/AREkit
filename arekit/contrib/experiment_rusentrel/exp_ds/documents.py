from arekit.common.experiment.api.ops_doc import DocumentOperations


class RuAttitudesDocumentOperations(DocumentOperations):

    def __init__(self, exp_ctx, ru_attitudes):
        assert(isinstance(ru_attitudes, dict))
        super(RuAttitudesDocumentOperations, self).__init__(exp_ctx=exp_ctx)
        self.__ru_attitudes = ru_attitudes

    def get_doc(self, doc_id):
        assert(isinstance(doc_id, int))
        return self.__ru_attitudes[doc_id]
