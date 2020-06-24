from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.base import BaseExperiment
from arekit.common.experiment.opinions import extract_text_opinions
from arekit.common.labels.base import Label
from arekit.common.linked.text_opinions.collection import LinkedTextOpinionCollection
from arekit.common.dataset.text_opinions.enums import EntityEndType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.collection import ParsedNewsCollection
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.text_opinion import TextOpinion


class OpinionProvider(object):
    """
    TextOpinion iterator + balancing.
    """

    def __init__(self, data_type, text_opinions, parsed_news_collection):
        assert(isinstance(data_type, DataType))
        assert(isinstance(text_opinions, LinkedTextOpinionCollection))
        assert(isinstance(parsed_news_collection, ParsedNewsCollection))
        self.__text_opinions = text_opinions
        self.__data_type = data_type
        self.__parsed_news_collection = parsed_news_collection
        self.__text_opinion_helper = TextOpinionHelper(parsed_news_collection)

    @classmethod
    def from_experiment(cls, experiment, data_type):
        assert(isinstance(experiment, BaseExperiment))

        pnc = experiment.create_parsed_collection(
            data_type=data_type,
            parse_frame_variants=False)

        assert(isinstance(pnc, ParsedNewsCollection))

        text_opinions = extract_text_opinions(
            experiment=experiment,
            data_type=data_type,
            terms_per_context=50,
            iter_doc_ids=pnc.iter_news_ids(),
            text_opinion_helper=TextOpinionHelper(pnc))

        return cls(data_type=data_type,
                   text_opinions=text_opinions,
                   parsed_news_collection=pnc)

    def opinions_count(self):
        return len(self.__text_opinions)

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
        parsed_news = self.__parsed_news_collection.get_by_news_id(text_opinion.NewsID)
        assert(isinstance(parsed_news, ParsedNews))

        return parsed_news, s_ind

    def get_entity_value(self, text_opinion, end_type):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(end_type, EntityEndType))

        return self.__text_opinion_helper.extract_entity_value(text_opinion=text_opinion,
                                                               end_type=end_type)
