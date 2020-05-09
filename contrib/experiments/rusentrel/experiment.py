from arekit.common.experiment.formats.cv_based.experiment import CVBasedExperiment
from arekit.contrib.experiments.rusentrel.documents import RuSentrelDocumentOperations
from arekit.contrib.experiments.rusentrel.opinions import RuSentrelOpinionOperations
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils


class RuSentRelExperiment(CVBasedExperiment):
    """
    Represents Input interface for NeuralNetwork ctx
    Now exploited (treated) as an input interface only
    """

    def __init__(self, data_io, prepare_model_root):

        opin_ops = RuSentrelOpinionOperations(
            data_io=data_io,
            annot_name=self.NeutralAnnotator.AnnotatorName,
            rusentrel_news_ids=set(RuSentRelIOUtils.iter_collection_indices()))

        doc_ops = RuSentrelDocumentOperations(data_io=data_io)

        super(RuSentRelExperiment, self).__init__(
            data_io=data_io,
            opin_ops=opin_ops,
            doc_ops=doc_ops,
            prepare_model_root=prepare_model_root)

