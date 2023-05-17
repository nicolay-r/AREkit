from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.opinions.annot.algo_based import AlgorithmBasedOpinionAnnotator


class AlgorithmBasedTextOpinionAnnotator(AlgorithmBasedOpinionAnnotator):
    """ This class represent a wrap over TextOpinionAnnotator
        and allows to perform a conversion into TextOpinions
    """

    def __init__(self, value_to_group_id_func, annot_algo, create_empty_collection_func,
                 get_doc_existed_opinions_func=None):
        """ get_doc_existed_opinions_func: func or None
                function that provides existed opinions for a document;
                if None, then we consider an absence of the existed document-level opinions.
        """
        assert(callable(value_to_group_id_func))
        super(AlgorithmBasedTextOpinionAnnotator, self).__init__(
            annot_algo=annot_algo,
            create_empty_collection_func=create_empty_collection_func,
            get_doc_existed_opinions_func=get_doc_existed_opinions_func)
        self.__value_to_group_id_func = value_to_group_id_func

    def __create_service(self, parsed_news):
        return ParsedNewsService(parsed_news=parsed_news, providers=[
            TextOpinionPairsProvider(self.__value_to_group_id_func)
        ])

    def annotate_collection(self, parsed_news):
        service = self.__create_service(parsed_news)
        topp = service.get_provider(TextOpinionPairsProvider.NAME)
        for opinion in super(AlgorithmBasedTextOpinionAnnotator, self).annotate_collection(parsed_news):
            for text_opinion in topp.iter_from_opinion(opinion):
                yield text_opinion
