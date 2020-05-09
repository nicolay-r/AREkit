from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils


class RuSentRelExperiment(CVBasedExperiment):
    """
    Represents a cv-based experiment over RuSentRel collection,
    which supports train/test separation.
    utilized in papers:
        https://link.springer.com/chapter/10.1007/978-3-030-23584-0_10
        https://wwww.easychair.org/publications/download/pQrC
    """

    def __init__(self, data_io, prepare_model_root):

        opin_ops = RuSentrelOpinionOperations(
            data_io=data_io,
            annot_name=self.NeutralAnnotator.AnnotatorName,
            rusentrel_news_ids=self.get_rusentrel_inds())

        doc_ops = RuSentrelDocumentOperations(data_io=data_io)

        super(RuSentRelExperiment, self).__init__(
            data_io=data_io,
            opin_ops=opin_ops,
            doc_ops=doc_ops,
            prepare_model_root=prepare_model_root)

    @staticmethod
    def get_rusentrel_inds():
        return set(RuSentRelIOUtils.iter_collection_indices())

