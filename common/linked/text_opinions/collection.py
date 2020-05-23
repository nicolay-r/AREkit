# -*- coding: utf-8 -*-
import logging
import collections
import cPickle as pickle

from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.common.text_opinions.collection import TextOpinionCollection


logger = logging.getLogger(__name__)


# TODO. Separate "Labeled": Create a different class that is
# TODO. considered to receive LinkedTextOpinionCollection instance in __init__
class LabeledLinkedTextOpinionCollection(TextOpinionCollection):
    """
    Describes text opinions with a position precision and forward connection

    Limitations:
    Not it represents IN-MEMORY implementation.
    Therefore it is supposed not so large amount of linked_text_opinions.
    """

    NO_NEXT_OPINION = None

    def __init__(self):
        super(LabeledLinkedTextOpinionCollection, self).__init__(text_opinions=[])

        # list describes that has i'th relation continuation in text.
        self.__next_opinion_id = []
        # provides original label by text_opinion_id
        # TODO. Relate to labeling
        self.__text_opinion_labels = []
        # labeling defined
        self.__labels_defined = []

    def try_add_linked_text_opinions(self,
                                     linked_text_opinions,
                                     check_opinion_correctness):
        """
        linked_text_opinions: iterable
            enumeration of text_opinions which is considered to be related to a certain opinion
            (in terms of Obj, Subj).
        check_opinion_correctness: bool
        """
        assert(isinstance(linked_text_opinions, collections.Iterable))
        assert(callable(check_opinion_correctness))

        discarded = 0
        registered_at_least_one = False
        for index, text_opinion in enumerate(linked_text_opinions):
            assert(isinstance(text_opinion, TextOpinion))
            assert(text_opinion.TextOpinionID is None)
            assert(text_opinion.Owner is None)

            registered = TextOpinion.create_copy(text_opinion)
            registered.set_owner(self)
            registered.set_text_opinion_id(len(self))

            self.register_text_opinion(registered)

            if not check_opinion_correctness(registered):
                discarded += 1
                self.remove_last_registered_text_opinion()
                del registered
            else:
                registered_at_least_one = True

        if registered_at_least_one:
            self.set_none_for_last_text_opinion()

        return discarded

    # TODO. Private
    def set_none_for_last_text_opinion(self):
        self.__next_opinion_id[-1] = self.NO_NEXT_OPINION

    def register_text_opinion(self, text_opinion):
        assert(isinstance(text_opinion, TextOpinion))
        super(LabeledLinkedTextOpinionCollection, self).register_text_opinion(text_opinion)
        self.__next_opinion_id.append(text_opinion.TextOpinionID + 1)
        # TODO. Relate to labeling
        self.__text_opinion_labels.append(text_opinion.Sentiment)
        self.__labels_defined.append(True)

    def remove_last_registered_text_opinion(self):
        super(LabeledLinkedTextOpinionCollection, self).remove_last_registered_text_opinion()
        del self.__next_opinion_id[-1]
        # TODO. Relate to labeling
        del self.__text_opinion_labels[-1]
        del self.__labels_defined[-1]

    # region public `check` methods

    # TODO. Related to Labeling
    def check_all_text_opinions_has_labels(self):
        return not (False in self.__labels_defined)

    # TODO. Related to Labeling
    def check_all_text_opinions_without_labels(self):
        return not (True in self.__labels_defined)

    # endregion

    # TODO. Related to Labeling
    def apply_label(self, label, text_opinion_id):
        assert(isinstance(text_opinion_id, int))

        text_opinion = self[text_opinion_id]
        assert(isinstance(text_opinion, TextOpinion))

        if self.__labels_defined[text_opinion_id] is not False:
            # assert(text_opinion.Sentiment == label)
            if text_opinion.Sentiment != label:
                logger.info("[Warning]: labels collision detected!")
            return

        text_opinion.set_label(label)
        self.__labels_defined[text_opinion_id] = True

    # region public `get` methods

    # TODO. Remove
    # TODO. Remove
    # TODO. Remove
    def get_labels_defined_count(self):
        return self.__labels_defined.count(True)

    # TODO. Remove
    # TODO. Remove
    # TODO. Remove
    def get_original_label(self, text_opinion_id):
        assert(isinstance(text_opinion_id, int))
        return self.__text_opinion_labels[text_opinion_id]

    # endregion

    # TODO. Related to Labeling
    def reset_labels(self):
        for text_opinion in self:
            text_opinion.set_label(self.__text_opinion_labels[text_opinion.TextOpinionID])
        self.__labels_defined = [False] * len(self.__labels_defined)

    # region public serialization methods

    def save(self, pickle_filepath):
        pickle.dump(self, open(pickle_filepath, 'wb'))

    @classmethod
    def load(cls, pickle_filepath):
        return pickle.load(open(pickle_filepath, 'rb'))

    # endregion

    # region iter methods

    def iter_wrapped_linked_text_opinions(self, news_id=None):
        assert(isinstance(news_id, int) or news_id is None)

        for linked_wrap in self.__iter_by_linked_text_opinions():
            if news_id is not None and linked_wrap.RelatedNewsID != news_id:
                continue

            yield linked_wrap

    def __iter_by_linked_text_opinions(self):
        linked_opinions = []
        for index, text_opinion in enumerate(self):
            linked_opinions.append(text_opinion)
            if self.__next_opinion_id[index] == self.NO_NEXT_OPINION:
                yield LinkedTextOpinionsWrapper(linked_text_opinions=linked_opinions)
                linked_opinions = []

    # endregion
