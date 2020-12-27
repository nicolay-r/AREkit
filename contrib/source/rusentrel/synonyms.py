from arekit.common.synonyms import SynonymsCollection
from arekit.processing.lemmatization.base import Stemmer


class StemmerBasedSynonymCollection(SynonymsCollection):

    def __init__(self, iter_group_values_lists, stemmer, is_read_only, debug):
        assert(isinstance(stemmer, Stemmer))
        self.__stemmer = stemmer
        super(StemmerBasedSynonymCollection, self).__init__(iter_group_values_lists=iter_group_values_lists,
                                                            is_read_only=is_read_only,
                                                            debug=debug)

    def create_synonym_id(self, value):
        # That may take a significant amount of time
        # especially when stemmer is a Yandex Mystem module.
        return self.__stemmer.lemmatize_to_str(value)

    def _create_internal_sid(self, value):
        # That may take a significant amount of time
        # especially when stemmer is a Yandex Mystem module.
        return self.__stemmer.lemmatize_to_str(value)
