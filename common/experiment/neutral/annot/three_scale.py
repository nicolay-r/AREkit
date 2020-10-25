import logging

from arekit.common.experiment.neutral.algo.default import DefaultNeutralAnnotationAlgorithm
from arekit.common.experiment.neutral.annot.base import BaseNeutralAnnotator
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.neutral.annot.labels_fmt import ThreeScaleLabelsFormatter
from arekit.common.opinions.formatter import OpinionCollectionsFormatter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ThreeScaleNeutralAnnotator(BaseNeutralAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    Three scale classification task.
    """

    IGNORED_ENTITY_VALUES = [u"author", u"unknown"]

    def __init__(self, distance_in_terms_between_bounds):
        super(ThreeScaleNeutralAnnotator, self).__init__()
        self.__algo = None
        self.__labels_fmt = ThreeScaleLabelsFormatter()
        self.__distance_in_terms_between_bounds = distance_in_terms_between_bounds

    # region private methods

    def __create_opinions_for_extraction(self, doc_id, data_type):
        assert(isinstance(data_type, DataType))

        news = self._DocOps.read_news(doc_id=doc_id)
        opinions = self._OpinOps.read_etalon_opinion_collection(doc_id=doc_id)

        if self.__algo is None:
            logger.info("Setup default annotation algorithm ...")
            self.__init_neutral_annotation_algo()

        collection = self.__algo.make_neutrals(
            news_id=doc_id,
            entities_collection=news.DocEntities,
            sentiment_opinions=opinions if data_type == DataType.Train else None)

        return collection

    def __init_neutral_annotation_algo(self):
        """
        Note: This operation might take a lot of time, as it assumes to perform news parsing.
        """
        self.__algo = DefaultNeutralAnnotationAlgorithm(
            synonyms=self._SynonymsCollection,
            iter_parsed_news=self._DocOps.iter_parsed_news(doc_inds=self._iter_doc_its_to_annotate()),
            dist_in_terms_bound=self.__distance_in_terms_between_bounds,
            ignored_entity_values=self.IGNORED_ENTITY_VALUES)

    # endregion

    def create_collection(self, data_type, opinion_formatter):
        assert(isinstance(data_type, DataType))
        assert(isinstance(opinion_formatter, OpinionCollectionsFormatter))

        for doc_id, filepath in self._iter_docs(data_type):
            collection = self.__create_opinions_for_extraction(doc_id=doc_id,
                                                               data_type=data_type)

            opinion_formatter.save_to_file(collection=collection,
                                           filepath=filepath,
                                           labels_formatter=self.__labels_fmt)



