import logging

from arekit.common.experiment.neutral.algo.default import DefaultNeutralAnnotationAlgorithm
from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator
from arekit.common.experiment.data_type import DataType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ThreeScaleNeutralAnnotator(BaseNeutralAnnotator):
    """ For three scale classification task.
    """

    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self, distance_in_terms_between_bounds):
        super(ThreeScaleNeutralAnnotator, self).__init__()
        self.__algo = None
        self.__distance_in_terms_between_bounds = distance_in_terms_between_bounds

    @property
    def Name(self):
        return u"annot-3-scale"

    # region private methods

    def _before_neutral_collections_iter(self, doc_ids_to_annot):
        self.__init_neutral_annotation_algo(doc_ids_to_annot)

    def _create_collection_core(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        news = self._DocOps.read_news(doc_id=doc_id)
        opinions = self._OpinOps.read_etalon_opinion_collection(doc_id=doc_id)

        neutral_opins_it = self.__algo.iter_neutral_opinions(
            news_id=doc_id,
            entities_collection=news.get_entities_collection(),
            sentiment_opinions=opinions if data_type == DataType.Train else None)

        return self._OpinOps.create_opinion_collection(neutral_opins_it)

    def __init_neutral_annotation_algo(self, doc_ids_to_annot):
        """
        Note: This operation might take a lot of time, as it assumes to perform news parsing.
        """
        self.__algo = DefaultNeutralAnnotationAlgorithm(
            iter_parsed_news=self._DocOps.iter_parsed_news(doc_inds=doc_ids_to_annot),
            dist_in_terms_bound=self.__distance_in_terms_between_bounds,
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)

    # endregion

