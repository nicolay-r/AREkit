import logging

from arekit.common.experiment.neutral.algo.default import DefaultNeutralAnnotationAlgorithm
from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator
from arekit.common.experiment.data_type import DataType
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.opinions.collection import OpinionCollection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ThreeScaleNeutralAnnotator(BaseNeutralAnnotator):
    """ For three scale classification task.
    """

    name = u"annot-3-scale"
    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self, distance_in_terms_between_bounds):
        super(ThreeScaleNeutralAnnotator, self).__init__()
        self.__algo = DefaultNeutralAnnotationAlgorithm(
            dist_in_terms_bound=distance_in_terms_between_bounds,
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)

    @property
    def Name(self):
        return ThreeScaleNeutralAnnotator.name

    # region private methods

    def _create_collection_core(self, parsed_news, data_type):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(data_type, DataType))

        news = self._DocOps.read_news(doc_id=parsed_news.RelatedNewsID)
        opinions = self._OpinOps.read_etalon_opinion_collection(doc_id=parsed_news.RelatedNewsID)

        neutral_opins_it = self.__algo.iter_neutral_opinions(
            parsed_news=parsed_news,
            entities_collection=news.get_entities_collection(),
            sentiment_opinions=opinions if data_type == DataType.Train else None)

        collection = self._OpinOps.create_opinion_collection()
        assert(isinstance(collection, OpinionCollection))

        # Filling. Keep all the opinions without duplications.
        for opinion in neutral_opins_it:
            if collection.has_synonymous_opinion(opinion):
                continue
            collection.add_opinion(opinion)

        return collection

    # endregion

