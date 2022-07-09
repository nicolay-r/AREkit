from arekit.common.synonyms.base import SynonymsCollection
from arekit.common.text.stemmer import Stemmer


class StemmerBasedSynonymCollection(SynonymsCollection):

    def __init__(self, iter_group_values_lists, stemmer, is_read_only, debug):
        assert(isinstance(stemmer, Stemmer))
        self.__stemmer = stemmer
        super(StemmerBasedSynonymCollection, self).__init__(iter_group_values_lists=iter_group_values_lists,
                                                            is_read_only=is_read_only,
                                                            debug=debug)

    # region private methods

    def __create_sid(self, value):
        # That may take a significant amount of time
        # especially when stemmer is a Yandex Mystem module.
        return self.__stemmer.lemmatize_to_str(value)

    # endregion

    # region protected methods

    def _create_external_sid(self, value):
        return self.__create_sid(value)

    def _create_internal_sid(self, value):
        return self.__create_sid(value)

    # endregion
