import logging

from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator
from arekit.common.labels.base import NeutralLabel
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.experiment.data_type import DataType

logger = logging.getLogger(__name__)


class TwoScaleNeutralAnnotator(BaseNeutralAnnotator):
    """ For two scale classification task.
    """

    name = u"annot-2-scale"

    def __init__(self):
        super(TwoScaleNeutralAnnotator, self).__init__()

    @property
    def Name(self):
        return TwoScaleNeutralAnnotator.name

    # region static methods

    @staticmethod
    def __iter_opinion_as_neutral(collection):
        assert(isinstance(collection, OpinionCollection))

        for opinion in collection:
            yield Opinion(source_value=opinion.SourceValue,
                          target_value=opinion.TargetValue,
                          sentiment=NeutralLabel())

    def _create_collection_core(self, parsed_news, data_type):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(data_type, DataType))

        doc_id = parsed_news.RelatedNewsID
        neut_collection = self._OpinOps.create_opinion_collection()
        assert(isinstance(neut_collection, OpinionCollection))

        # We copy all the opinions from etalon collection
        # into neutral one with the replaced sentiment values.
        # as we treat such opinions as neutral one since only NeutralLabels
        # could be casted into correct string.
        for opinion in self._OpinOps.read_etalon_opinion_collection(doc_id):
            neut_collection.add_opinion(Opinion(source_value=opinion.SourceValue,
                                                target_value=opinion.TargetValue,
                                                sentiment=NeutralLabel()))

        return neut_collection


    # endregion

    # region public methods

    def serialize_missed_collections(self, data_type):

        if data_type == DataType.Train:
            return

        super(TwoScaleNeutralAnnotator, self).serialize_missed_collections(data_type)

    # endregion
