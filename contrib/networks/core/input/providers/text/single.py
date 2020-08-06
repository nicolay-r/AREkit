from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.contrib.networks.core.input.terms_mapping import StringWithEmbeddingNetworkTermMapping


class NetworkSingleTextProvider(BaseSingleTextProvider):
    """
    Performs iteration process over (string, embedding) term pairs.
    """

    def __init__(self, text_terms_mapper, pair_handling_func):
        assert(isinstance(text_terms_mapper, StringWithEmbeddingNetworkTermMapping))
        assert(callable(pair_handling_func))
        super(NetworkSingleTextProvider, self).__init__(text_terms_mapper=text_terms_mapper)

        self.__write_embedding_pair_func = pair_handling_func

    def _compose_text(self, sentence_terms):
        terms = []
        for pair in self._mapper.iter_mapped(sentence_terms):
            term, embedding = pair
            self.__write_embedding_pair_func(pair)
            terms.append(term)

        return self.TERMS_SEPARATOR.join(terms)

