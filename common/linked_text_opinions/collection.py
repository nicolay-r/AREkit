# -*- coding: utf-8 -*-
import cPickle as pickle
import collections
from core.common.linked_text_opinions.text_opinion import TextOpinion
from core.common.parsed_news.collection import ParsedNewsCollection


class LabeledLinkedTextOpinionCollection(object):
    """
    Describes text opinions with a position precision and forward connection

    Usage:
    It is possible to add linked_text_opinions from news using add_news_relations call

    Limitations:
    Not it represents IN-MEMORY implementation.
    Therefore it is supposed not so large amount of linked_text_opinions.
    """

    NO_NEXT_RELATION = None

    def __init__(self, parsed_news_collection):
        assert(isinstance(parsed_news_collection, ParsedNewsCollection))
        # list ExtractedRelations
        # TODO. In Base.
        self.__text_opinions = []
        # list describes that has i'th relation continuation in text.
        self.__next_opinion_id = []
        # provides original label by text_opinion_id
        self.__text_opinion_labels = []
        # labels defined
        self.__labels_defined = []
        # related collection of parsed news
        # TODO. In Base.
        self.__parsed_news_collection = parsed_news_collection

    # TODO. In Base.
    @property
    def RelatedParsedNewsCollection(self):
        return self.__parsed_news_collection

    # TODO. In base
    @property
    def TextOpinionCount(self):
        return len(self.__text_opinions)

    def add_text_opinions(self,
                          text_opinions,
                          check_opinion_correctness):
        assert(isinstance(text_opinions, collections.Iterable))
        assert(callable(check_opinion_correctness))

        missed = 0
        for index, text_opinion in enumerate(text_opinions):
            assert(isinstance(text_opinion, TextOpinion))
            assert(text_opinion.TextOpinionID is None)

            if not check_opinion_correctness(text_opinion):
                missed += 1
                continue

            text_opinion.set_text_opinion_id(len(self.__text_opinions))

            self.register_text_opinion(text_opinion)

        self.set_none_for_last_text_opinion()
        return missed

    def set_none_for_last_text_opinion(self):
        self.__next_opinion_id[-1] = self.NO_NEXT_RELATION

    def register_text_opinion(self, text_opinion):
        assert(isinstance(text_opinion, TextOpinion))
        self.__text_opinions.append(text_opinion)
        self.__next_opinion_id.append(text_opinion.TextOpinionID + 1)
        self.__text_opinion_labels.append(text_opinion.Sentiment)
        self.__labels_defined.append(True)

    def check_all_relations_has_labels(self):
        return not (False in self.__labels_defined)

    def check_all_relations_without_labels(self):
        return not (True in self.__labels_defined)

    def apply_label(self, label, text_opinion_id):
        assert(isinstance(text_opinion_id, int))

        if self.__labels_defined[text_opinion_id] is not False:
            assert(self.__text_opinions[text_opinion_id].Label == label)
            return

        self.__text_opinions[text_opinion_id].set_label(label)
        self.__labels_defined[text_opinion_id] = True

    def get_original_label(self, text_opinion_id):
        assert(isinstance(text_opinion_id, int))
        return self.__text_opinion_labels[text_opinion_id]

    def reset_labels(self):
        for opinion in self.__text_opinions:
            opinion.set_label(self.__text_opinion_labels[opinion.TextOpinionID])
        self.__labels_defined = [False] * len(self.__labels_defined)

    def save(self, pickle_filepath):
        pickle.dump(self, open(pickle_filepath, 'wb'))

    @classmethod
    def load(cls, pickle_filepath):
        return pickle.load(open(pickle_filepath, 'rb'))

    def __iter_by_linked_text_opinions(self):
        lst = []
        for index, text_opinion in enumerate(self.__text_opinions):
            lst.append(text_opinion)
            if self.__next_opinion_id[index] == self.NO_NEXT_RELATION:
                yield lst
                lst = []

    def iter_by_linked_text_opinions(self):
        return self.__iter_by_linked_text_opinions()

    def iter_by_linked_text_opinion_groups(self, group_size):
        assert(isinstance(group_size, int))
        group = []
        for index, linked_opinions in enumerate(self.__iter_by_linked_text_opinions()):
            group.append(linked_opinions)
            if len(group) == group_size:
                yield group
                group = []

    def __iter__(self):
        for text_opinion in self.__text_opinions:
            yield text_opinion

    def __len__(self):
        return len(self.__text_opinions)
