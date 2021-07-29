from arekit.common.news.entities_parser import BaseEntitiesParser
from arekit.contrib.source.rusentrel.sentence import RuSentRelSentence


class RuSentRelTextEntitiesParser(BaseEntitiesParser):

    def __init__(self):
        super(RuSentRelTextEntitiesParser, self).__init__()
        self.__sentence_text = None
        self.__text_subs_iter_func = None

    def _before_parsing(self, sentence):
        assert(isinstance(sentence, RuSentRelSentence))

        # We caching actual sentence text (string)
        self.__sentence_text = sentence.Text

        # We provide substitutions from the related text string.
        self.__text_subs_iter_func = lambda: sentence.iter_entity_with_local_bounds()

    def _get_sentence_length(self):
        return len(self.__sentence_text)

    def _iter_subs_values_with_bounds(self):
        return self.__text_subs_iter_func()

    def _iter_part(self, from_index, to_index):
        yield self.__sentence_text[from_index:to_index]
