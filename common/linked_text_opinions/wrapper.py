from arekit.common.text_opinions.text_opinion import TextOpinion


class LinkedTextOpinionsWrapper(object):
    """
    Wrapper for a list of text opinions
    """

    def __init__(self, linked_text_opinions):
        assert(isinstance(linked_text_opinions, list))
        self.__linked_opinions = linked_text_opinions

    @property
    def FirstOpinion(self):
        text_opinion = self.__linked_opinions[0]
        assert(isinstance(text_opinion, TextOpinion))
        return text_opinion

    @property
    def RelatedNewsID(self):
        return self.FirstOpinion.NewsID

    def get_linked_sentiment(self):
        return self.FirstOpinion.Sentiment

    def get_prior_opinion_by_index(self, index):
        assert(isinstance(index, int))
        return self.__linked_opinions[index - 1] if index > 0 else None

    def get_by_index(self, index):
        assert(isinstance(index, int))
        return self.__linked_opinions[index]

    def __iter__(self):
        for text_opinion in self.__linked_opinions:
            yield text_opinion

    def __len__(self):
        return len(self.__linked_opinions)

