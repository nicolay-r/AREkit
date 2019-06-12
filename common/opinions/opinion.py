from core.evaluation.labels import Label
from core.common.synonyms import SynonymsCollection


# TODO. To /common/collection.py
class Opinion(object):
    """ Source opinion description
    """

    def __init__(self, value_left, value_right, sentiment):
        assert(isinstance(value_left, unicode))
        assert(isinstance(value_right, unicode))
        assert(isinstance(sentiment, Label))
        assert(',' not in value_left)
        assert(',' not in value_right)
        self.value_left = value_left.lower()
        self.value_right = value_right.lower()
        self.sentiment = sentiment
        self.__tag = None

    @property
    def Tag(self):
        return self.__tag

    def set_tag(self, value):
        self.__tag = value

    def create_synonym_id(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return u"{}_{}".format(
            synonyms.get_synonym_group_index(self.value_left),
            synonyms.get_synonym_group_index(self.value_right))

    def has_synonym_for_left(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return synonyms.has_synonym(self.value_left)

    def has_synonym_for_right(self, synonyms):
        assert(isinstance(synonyms, SynonymsCollection))
        return synonyms.has_synonym(self.value_right)
