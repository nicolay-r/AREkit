import logging

from arekit.common.docs.parsed.base import ParsedDocument
from arekit.common.opinions.annot.algo.base import BaseOpinionAnnotationAlgorithm
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.common.opinions.collection import OpinionCollection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AlgorithmBasedOpinionAnnotator(BaseOpinionAnnotator):
    """ Algorithm-based annotator
    """

    def __init__(self, annot_algo, create_empty_collection_func, get_doc_existed_opinions_func=None):
        """ create_empty_collection_func: func
                function that creates an empty OpinionCollection
            get_doc_existed_opinions_func: func or None
                function that provides existed opinions for a document;
                if None, then we consider an absence of the existed document-level opinions.
        """
        assert(isinstance(annot_algo, BaseOpinionAnnotationAlgorithm))
        assert(callable(get_doc_existed_opinions_func) or get_doc_existed_opinions_func is None)
        super(AlgorithmBasedOpinionAnnotator, self).__init__()

        self.__annot_algo = annot_algo
        self.__create_empty_collection_func = create_empty_collection_func
        self.__get_existed_opinions_func = (lambda _: None) \
            if get_doc_existed_opinions_func is None else get_doc_existed_opinions_func

    # region private methods

    def _annot_collection_core(self, parsed_doc):
        assert(isinstance(parsed_doc, ParsedDocument))

        opinions = self.__get_existed_opinions_func(parsed_doc.RelatedDocID)
        assert(isinstance(opinions, OpinionCollection) or opinions is None)
        
        annotated_opinions_it = self.__annot_algo.iter_opinions(
            parsed_doc=parsed_doc, existed_opinions=opinions)

        collection = self.__create_empty_collection_func()
        assert(isinstance(collection, OpinionCollection))

        # Filling. Keep all the opinions without duplications.
        for opinion in annotated_opinions_it:
            if collection.has_synonymous_opinion(opinion):
                continue
            collection.add_opinion(opinion)

        return collection

    # endregion

