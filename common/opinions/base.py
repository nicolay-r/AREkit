from arekit.common.labels.base import Label
from arekit.common.synonyms import SynonymsCollection


class Opinion(object):
    """ Source opinion description
    """

    def __init__(self, source_value, target_value, sentiment):
        assert(isinstance(source_value, unicode))
        assert(isinstance(target_value, unicode))
        assert(isinstance(sentiment, Label))
        assert(',' not in source_value)  # Could be removed.
        assert(',' not in target_value)  # Could be removed.
        self.__source_value = source_value.lower()
        self.__target_value = target_value.lower()
        self.__sentiment = sentiment
        self.__tag = None

    # region properties

    @property
    def SourceValue(self):
        return self.__source_value

    @property
    def TargetValue(self):
        return self.__target_value

    @property
    def Sentiment(self):
        return self.__sentiment

    @property
    def Tag(self):
        return self.__tag

    # endregion

    # region public methods

    def set_tag(self, value):
        self.__tag = value

    def create_synonym_id(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return u"{}_{}".format(
            synonyms.get_synonym_group_index(self.__source_value),
            synonyms.get_synonym_group_index(self.__target_value))

    def has_synonym_for_source(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return synonyms.contains_synonym_value(self.__source_value)

    def has_synonym_for_target(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return synonyms.contains_synonym_value(self.__target_value)

    # endregion
