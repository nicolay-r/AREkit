from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.contrib.networks.input.terms_mapping import StringWithEmbeddingNetworkTermMapping


class NetworkSingleTextProvider(BaseSingleTextProvider):
    """
    Performs iteration process over (string, embedding) term pairs.
    """

    def __init__(self, text_terms_mapper, pair_handling_func):
        assert(isinstance(text_terms_mapper, StringWithEmbeddingNetworkTermMapping))
        assert(callable(pair_handling_func))
        super(NetworkSingleTextProvider, self).__init__(text_terms_mapper=text_terms_mapper)
        self.__write_embedding_pair_func = pair_handling_func

    def _mapped_data_to_str(self, m_data):
        # In this case, m_term consist of
        # 1. Term;
        # 2. Embedding.
        term, _ = m_data
        return term

    def _handle_mapped_data(self, m_data):
        self.__write_embedding_pair_func(m_data)
