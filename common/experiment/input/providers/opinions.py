import collections

from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.opinions import extract_text_opinions
from arekit.common.labels.base import Label
from arekit.common.linked.text_opinions.collection import LinkedTextOpinionCollection
from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.base import TextOpinion


class OpinionProvider(object):
    """
    TextOpinion iterator + balancing.
    """

    def __init__(self, text_opinions, text_opinion_helper):
        assert(isinstance(text_opinions, LinkedTextOpinionCollection))
        assert(isinstance(text_opinion_helper, TextOpinionHelper))
        self.__text_opinions = text_opinions
        self.__text_opinion_helper = text_opinion_helper

    @classmethod
    def from_experiment(cls, doc_ops, opin_ops, data_type, iter_news_ids, terms_per_context, text_opinion_helper):
        assert(isinstance(doc_ops, DocumentOperations))
        assert(isinstance(opin_ops, OpinionOperations))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(iter_news_ids, collections.Iterable))
        assert(isinstance(text_opinion_helper, TextOpinionHelper))

        text_opinions = extract_text_opinions(
            doc_ops=doc_ops,
            opin_ops=opin_ops,
            data_type=data_type,
            terms_per_context=terms_per_context,
            iter_doc_ids=iter_news_ids,
            text_opinion_helper=text_opinion_helper)

        return cls(text_opinions=text_opinions,
                   text_opinion_helper=text_opinion_helper)

    # region private methods

    def __iter_linked_opinons(self, label, count):
        while count > 0:
            for linked_wrap in self.__text_opinions.iter_wrapped_linked_text_opinions():

                if linked_wrap.get_linked_label() != label:
                    continue

                yield linked_wrap
                count -= 1

                if count == 0:
                    break

    def __iter_balanced(self, supported_labels):
        assert(isinstance(supported_labels, list))

        counts = {}
        for label in supported_labels:
            assert(isinstance(label, Label))
            counts[label] = 0

        for linked_wrap in self.__text_opinions.iter_wrapped_linked_text_opinions():
            counts[linked_wrap.get_linked_label()] += 1

            yield linked_wrap

        top = max(counts.values())

        for label in counts.iterkeys():
            if counts[label] == 0:
                continue
            left = top-counts[label]
            for linked_opinions in self.__iter_linked_opinons(label=label, count=left):
                yield linked_opinions

    # endregion

    def iter_linked_opinion_wrappers(self, balance, supported_labels):
        assert(isinstance(balance, bool))
        assert(isinstance(supported_labels, list) or supported_labels is None)

        linked_opinion_wrap_it = self.__text_opinions.iter_wrapped_linked_text_opinions() \
            if not balance else self.__iter_balanced(supported_labels=supported_labels)

        for linked_wrap in linked_opinion_wrap_it:
            yield linked_wrap

    def get_opinion_location(self, text_opinion):
        assert(isinstance(text_opinion, TextOpinion))

        # Determining text_a by text_opinion end.
        s_ind = self.__text_opinion_helper.extract_entity_position(
            text_opinion=text_opinion,
            end_type=EntityEndType.Source,
            position_type=TermPositionTypes.SentenceIndex)

        # Extract specific document by text_opinion.NewsID
        parsed_news = self.__text_opinion_helper.get_related_news(text_opinion)

        return parsed_news, s_ind

    def get_entity_value(self, text_opinion, end_type):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(end_type, EntityEndType))

        return self.__text_opinion_helper.extract_entity_value(text_opinion=text_opinion,
                                                               end_type=end_type)
