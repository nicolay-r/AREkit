from arekit.common.labels.base import Label
from arekit.common.opinions.enums import OpinionEndTypes
from arekit.common.synonyms.base import SynonymsCollection


class Opinion(object):
    """ Source opinion description
    """

    def __init__(self, source_value, target_value, sentiment):
        assert(isinstance(source_value, str))
        assert(isinstance(target_value, str))
        assert(isinstance(sentiment, Label))
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

    def __get_end_synonym_inds(self, synonyms):
        s_ind = synonyms.get_synonym_group_index(self.__source_value)
        t_ind = synonyms.get_synonym_group_index(self.__target_value)
        return s_ind, t_ind

    # region public methods

    def get_value(self, end_type):
        assert(isinstance(end_type, OpinionEndTypes))

        if end_type == OpinionEndTypes.Source:
            return self.SourceValue

        if end_type == OpinionEndTypes.Target:
            return self.TargetValue

        raise Exception("Unknown end_type='{e_type}'".format(e_type=end_type))

    def set_tag(self, value):
        self.__tag = value

    def is_loop(self, synonyms):
        s_ind, t_ind = self.__get_end_synonym_inds(synonyms)
        return s_ind == t_ind

    def create_synonym_id(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        s_ind, t_ind = self.__get_end_synonym_inds(synonyms)
        return "{}_{}".format(s_ind, t_ind)

    def has_synonym_for_end(self, synonyms, end_type):
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(end_type, OpinionEndTypes))
        return synonyms.contains_synonym_value(self.get_value(end_type))

    # endregion
