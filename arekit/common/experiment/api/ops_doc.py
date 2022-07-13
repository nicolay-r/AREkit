from arekit.common.experiment.api.ctx_base import ExperimentContext


class DocumentOperations(object):
    """ Provides operations with documents
    """

    def __init__(self, exp_ctx):
        assert(isinstance(exp_ctx, ExperimentContext) or exp_ctx is None)
        self._exp_ctx = exp_ctx

    def get_doc(self, doc_id):
        raise NotImplementedError()
