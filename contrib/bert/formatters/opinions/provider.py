from arekit.common.experiment.opinions import extract_text_opinions_and_parse_news
from arekit.common.labels.base import Label
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper


class OpinionProvider(object):
    """
    TextOpinion iterator + balancing.
    """

    def __init__(self, data_type, text_opinions):
        assert(isinstance(data_type, unicode))
        self.__text_opinions = text_opinions
        self.__data_type = data_type

    @classmethod
    def from_experiment(cls, experiment, data_type):
        text_opinions = extract_text_opinions_and_parse_news(
            experiment=experiment,
            data_type=data_type,
            terms_per_context=50)

        return cls(data_type=data_type,
                   text_opinions=text_opinions)

    def opinions_count(self):
        return len(self.__text_opinions)

    # region private methods

    def __iter_linked_opinons(self, label, count):
        while count > 0:
            for linked_wrap in self.__text_opinions.iter_wrapped_linked_text_opinions():

                if linked_wrap.get_linked_sentiment() != label:
                    continue

                yield linked_wrap
                count -= 1

                if count == 0:
                    break

    def __iter_balanced(self):
        counts = {}
        for label in Label._get_supported_labels():
            counts[label] = 0

        for linked_wrap in self.__text_opinions.iter_wrapped_linked_text_opinions():
            counts[linked_wrap.get_linked_sentiment()] += 1

            yield linked_wrap

        top = max(counts.values())

        for label in counts.iterkeys():
            if counts[label] == 0:
                continue
            left = top-counts[label]
            for linked_opinions in self.__iter_linked_opinons(label=label, count=left):
                yield linked_opinions

    # endregion

    def iter_linked_opinion_wrappers(self, balance):
        assert(isinstance(balance, bool))

        linked_opinion_wrap_it = self.__text_opinions.iter_wrapped_linked_text_opinions() \
            if not balance else self.__iter_balanced()

        for linked_wrap in linked_opinion_wrap_it:
            yield linked_wrap

    def get_opinion_location(self, text_opinion):

        pnc = self.__text_opinions.RelatedParsedNewsCollection
        assert(isinstance(pnc, ParsedNewsCollection))

        # Determining text_a by text_opinion end.
        s_ind = TextOpinionHelper.extract_entity_sentence_index(
            text_opinion=text_opinion,
            end_type=EntityEndType.Source)

        # Extract specific document by text_opinion.NewsID
        pn = pnc.get_by_news_id(text_opinion.NewsID)
        assert(isinstance(pn, ParsedNews))

        return pn, s_ind
