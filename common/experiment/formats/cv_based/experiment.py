from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.formats.cv_based.documents import CVBasedDocumentOperations
from arekit.common.experiment.formats.cv_based.opinions import CVBasedOpinionOperations


class CVBasedExperiment(BaseExperiment):

    def __init__(self,
                 data_io,
                 opin_ops,
                 doc_ops,
                 prepare_model_root):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(opin_ops, CVBasedOpinionOperations))
        assert(isinstance(doc_ops, CVBasedDocumentOperations))

        super(CVBasedExperiment, self).__init__(
            data_io=data_io,
            opin_operation=opin_ops,
            doc_operations=doc_ops,
            prepare_model_root=prepare_model_root)
