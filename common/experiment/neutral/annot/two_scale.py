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

    def __init__(self):
        super(TwoScaleNeutralAnnotator, self).__init__()

    @property
    def Name(self):
        return u"annot-2-scale"

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

        # TODO. Extract opinions from news.
        # TODO. exp_io.read_parsed_news.
        # TODO. exp_io.read_etalon_opinion_collection()

        doc_id = parsed_news.RelatedNewsID
        collection = self._OpinOps.read_etalon_opinion_collection(doc_id)
        return self._OpinOps.create_opinion_collection(
            opinions=list(self.__iter_opinion_as_neutral(collection)))

    # endregion

    # region public methods

    def serialize_missed_collections(self, data_type):

        if data_type == DataType.Train:
            return

        super(TwoScaleNeutralAnnotator, self).serialize_missed_collections(data_type)

    # endregion
