from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.text_opinions.text_opinion import TextOpinion


class LinkedTextOpinionsWrapper(LinkedDataWrapper):

    def __init__(self, linked_text_opinions):
        super(LinkedTextOpinionsWrapper, self).__init__(linked_data=linked_text_opinions)

    @property
    def First(self):
        first = super(LinkedTextOpinionsWrapper, self).First
        assert(isinstance(first, TextOpinion))
        return first

    @property
    def RelatedNewsID(self):
        return self.First.NewsID

    def get_linked_label(self):
        return self.First.Sentiment

    def _get_data_label(self, item):
        assert(isinstance(item, TextOpinion))
        return item.Sentiment

    def get_prior_opinion_by_index(self, index):
        assert(isinstance(index, int))
        return self[index - 1] if index > 0 else None