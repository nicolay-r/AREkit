from arekit.common.linked.data import LinkedDataWrapper
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
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

    @staticmethod
    def _aggregate_by_first(item, label):
        assert(isinstance(item, TextOpinion))
        source = TextOpinionHelper.extract_entity_value(item, EntityEndType.Source)
        target = TextOpinionHelper.extract_entity_value(item, EntityEndType.Target)

        return Opinion(source_value=source,
                       target_value=target,
                       sentiment=label)

    def get_linked_label(self):
        return self.First.Sentiment

    def get_prior_opinion_by_index(self, index):
        assert(isinstance(index, int))
        return self[index - 1] if index > 0 else None
