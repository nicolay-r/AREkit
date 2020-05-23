from arekit.common.text_opinions.collection import TextOpinionCollection
from arekit.common.text_opinions.text_opinion import TextOpinion

from io_utils import logger


class LabeledCollection:
    """
    Collection provides labeling for TextOpinionCollection
    """

    def __init__(self, collection):
        assert(isinstance(collection, TextOpinionCollection))

        self.__collection = collection
        self.__original_labels = [text_opinion.Sentiment for text_opinion in collection]
        self.__labels_defined = None
        self.__reset_definitive_state(True)

    def get_unique_news_ids(self):
        return self.__collection.get_unique_news_ids()

    def iter_text_opinions(self):
        for text_opinion in self.__collection:
            yield text_opinion

    def apply_label(self, label, text_opinion_id):
        assert(isinstance(text_opinion_id, int))

        text_opinion = self.__collection[text_opinion_id]
        assert(isinstance(text_opinion, TextOpinion))

        if self.__labels_defined[text_opinion_id] is not False:
            # assert(text_opinion.Sentiment == label)
            if text_opinion.Sentiment != label:
                logger.info("[Warning]: labels collision detected!")
            return

        text_opinion.set_label(label)
        self.__labels_defined[text_opinion_id] = True

    def check_all_text_opinions_has_labels(self):
        return not (False in self.__labels_defined)

    def check_all_text_opinions_without_labels(self):
        return not (True in self.__labels_defined)

    def reset_labels(self):
        for text_opinion in self.__collection:
            label = self.__original_labels[text_opinion.TextOpinionID]
            text_opinion.set_label(label)
        self.__reset_definitive_state(False)

    def __reset_definitive_state(self, flag):
        assert(isinstance(flag, bool))
        self.__labels_defined = [flag] * len(self.__labels_defined)
