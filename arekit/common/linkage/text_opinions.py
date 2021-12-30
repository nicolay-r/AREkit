from arekit.common.linkage.base import LinkedDataWrapper
from arekit.common.text_opinions.base import TextOpinion


class TextOpinionsLinkage(LinkedDataWrapper):

    def __init__(self, text_opinions_it):
        super(TextOpinionsLinkage, self).__init__(linked_data=text_opinions_it)
        self.__tag = None

    def set_tag(self, value):
        self.__tag = value

    @property
    def First(self):
        first = super(TextOpinionsLinkage, self).First
        assert(isinstance(first, TextOpinion))
        return first

    @property
    def Tag(self):
        return self.__tag

    @property
    def RelatedDocID(self):
        return self.First.DocID

    def get_linked_label(self):
        return self.First.Sentiment

    def _get_data_label(self, item):
        assert(isinstance(item, TextOpinion))
        return item.Sentiment
