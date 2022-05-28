import logging

from arekit.common.experiment.annot.base import BaseAnnotator
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.experiment.data_type import DataType
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel

logger = logging.getLogger(__name__)


class TwoScaleTaskAnnotator(BaseAnnotator):
    """ For two scale classification task.
    """

    def __init__(self):
        super(TwoScaleTaskAnnotator, self).__init__()

    # region static methods

    def _annot_collection_core(self, parsed_news, data_type, opin_ops):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(data_type, DataType))

        doc_id = parsed_news.RelatedDocID
        neut_collection = opin_ops.create_opinion_collection()
        assert(isinstance(neut_collection, OpinionCollection))

        # We copy all the opinions from etalon collection
        # into neutral one with the replaced sentiment values.
        # as we treat such opinions as neutral one since only NeutralLabels
        # could be casted into correct string.
        for opinion in opin_ops.get_etalon_opinion_collection(doc_id):
            neut_collection.add_opinion(Opinion(source_value=opinion.SourceValue,
                                                target_value=opinion.TargetValue,
                                                sentiment=ExperimentNeutralLabel()))

        return neut_collection

    # endregion

    # region public methods

    def annotate_collection(self, data_type, parsed_news, opin_ops):

        if data_type == DataType.Train:
            # Return empty collection.
            return opin_ops.create_opinion_collection()

        super(TwoScaleTaskAnnotator, self).annotate_collection(
            data_type, parsed_news=parsed_news, opin_ops=opin_ops)

    # endregion
