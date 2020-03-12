from arekit.contrib.experiments.neutral.annot.rusentrel_three_scale import RuSentRelThreeScaleNeutralAnnotator
from arekit.networks.data_type import DataType


class RuSentRelTwoScaleNeutralAnnotator(RuSentRelThreeScaleNeutralAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    For two scale classification task.
    """

    def __init__(self, experiments_io, stemmer, create_synonyms_collection):
        super(RuSentRelTwoScaleNeutralAnnotator, self).__init__(
            experiments_io=experiments_io,
            stemmer=stemmer,
            create_synonyms_collection=create_synonyms_collection)

    def create(self, data_type):
        assert(isinstance(data_type, unicode))

        if data_type == DataType.Train:
            return

        # TODO. Implement.
