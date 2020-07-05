from arekit.common.experiment.input.providers.text.single import BaseSingleTextProvider
from arekit.networks.input.terms_mapping import EmbeddedTermMapping


class SingleTextProvider(BaseSingleTextProvider):

    def __init__(self, text_terms_mapper, write_embedding_pair_func):
        assert(isinstance(text_terms_mapper, EmbeddedTermMapping))
        assert(callable(write_embedding_pair_func))
        super(SingleTextProvider, self).__init__(text_terms_mapper=text_terms_mapper)

        self.__write_embedding_pair_func = write_embedding_pair_func

    def _compose_text(self, sentence_terms):
        terms = []
        for pair in self._mapper.iter_mapped(sentence_terms):
            term, embedding = pair
            self.__write_embedding_pair_func(pair)
            terms.append(term)

        return self.TERMS_SEPARATOR.join(terms)

