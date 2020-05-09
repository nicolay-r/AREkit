from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.text_opinions.text_opinion import TextOpinion


class LinkedTextOpinionsWrapper(LinkedDataWrapper):

    def __init__(self, linked_text_opinions):
        super(LinkedTextOpinionsWrapper, self).__init__(linked_data=linked_text_opinions)

    @property
    def FirstOpinion(self):
        text_opinion = self[0]
        assert(isinstance(text_opinion, TextOpinion))
        return text_opinion

    @property
    def RelatedNewsID(self):
        return self.FirstOpinion.NewsID

    def get_linked_label(self):
        return self.FirstOpinion.Sentiment

    def get_prior_opinion_by_index(self, index):
        assert(isinstance(index, int))
        return self[index - 1] if index > 0 else None
