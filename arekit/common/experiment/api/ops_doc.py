from arekit.common.experiment.api.ctx_base import ExperimentContext


class DocumentOperations(object):
    """
    Provides operations with documents
    """

    def __init__(self, exp_ctx):
        assert(isinstance(exp_ctx, ExperimentContext) or exp_ctx is None)
        self._exp_ctx = exp_ctx

    # region abstract methods

    def get_doc(self, doc_id):
        raise NotImplementedError()

    def iter_tagget_doc_ids(self, tag):
        """ Document identifiers which are grouped by a particular tag.
        """
        raise NotImplementedError()

    # endregion

    # region public methods

    def iter_doc_ids(self, data_type):
        """ Provides a news indices, related to a particular `data_type`
        """
        data_types_splits = self._exp_ctx.DataFolding.fold_doc_ids_set()

        if data_type not in data_types_splits:
            return
            yield

        for doc_id in data_types_splits[data_type]:
            yield doc_id

    # endregion