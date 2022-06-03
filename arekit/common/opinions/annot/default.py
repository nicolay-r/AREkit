import logging

from arekit.common.experiment.data_type import DataType
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.opinions.annot.algo.base import BaseAnnotationAlgorithm
from arekit.common.opinions.annot.base import BaseAnnotator
from arekit.common.opinions.collection import OpinionCollection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DefaultAnnotator(BaseAnnotator):
    """ Algorithm-based annotator
    """

    def __init__(self, annot_algo, create_empty_collection_func, get_doc_etalon_opins_func):
        """
        create_empty_collection_func:
            function that creates an empty opinion collection
        get_doc_etalon_opins_func:
            function that provides etalon opinions for a document
        """
        assert(isinstance(annot_algo, BaseAnnotationAlgorithm))
        assert(callable(get_doc_etalon_opins_func))
        super(DefaultAnnotator, self).__init__()

        self.__annot_algo = annot_algo
        self.__get_doc_etalon_opins_func = get_doc_etalon_opins_func
        self.__create_empty_collection_func = create_empty_collection_func

    # region private methods

    def _annot_collection_core(self, parsed_news, data_type):
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(data_type, DataType))

        opinions = self.__get_doc_etalon_opins_func(parsed_news.RelatedDocID)

        annotated_opins_it = self.__annot_algo.iter_opinions(
            parsed_news=parsed_news,
            existed_opinions=opinions if data_type == DataType.Train else None)

        collection = self.__create_empty_collection_func()
        assert(isinstance(collection, OpinionCollection))

        # Filling. Keep all the opinions without duplications.
        for opinion in annotated_opins_it:
            if collection.has_synonymous_opinion(opinion):
                continue
            collection.add_opinion(opinion)

        return collection

    # endregion

