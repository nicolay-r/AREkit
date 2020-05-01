from arekit.common.experiment.opinions import extract_text_opinions
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
        text_opinions = extract_text_opinions(
            experiment=experiment,
            data_type=data_type,
            terms_per_context=50)

        return cls(data_type=data_type,
                   text_opinions=text_opinions)

    def opinions_count(self):
        return len(self.__text_opinions)

    @staticmethod
    def get_linked_sentiment(linked_opinions):
        return linked_opinions[0].Sentiment

    # region private methods

    def __iter_linked_with_sentiment(self):
        for linked_opinions in self.__text_opinions.iter_by_linked_text_opinions():
            label = self.get_linked_sentiment(linked_opinions)
            yield linked_opinions, label

    def __iter_linked_opinons(self, label, count):
        while count > 0:
            for linked_opinions, linked_label in self.__iter_linked_with_sentiment():

                if linked_label != label:
                    continue

                yield linked_opinions
                count -= 1

                if count == 0:
                    break

    def __iter_balanced(self):
        counts = {}
        for label in Label._get_supported_labels():
            counts[label] = 0

        for linked_opinions, label in self.__iter_linked_with_sentiment():
            counts[label] += 1

            yield linked_opinions

        top = max(counts.values())

        for label in counts.iterkeys():
            if counts[label] == 0:
                continue
            left = top-counts[label]
            for linked_opinions in self.__iter_linked_opinons(label=label, count=left):
                yield linked_opinions

    # endregion

    def iter_linked_opinions(self, balance):
        assert(isinstance(balance, bool))

        if not balance:
            for linked_opinions in self.__text_opinions.iter_by_linked_text_opinions():
                # TODO. Wrap
                yield linked_opinions
        else:
            for linked_opinions in self.__iter_balanced():
                # TODO. Wrap
                yield linked_opinions

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
