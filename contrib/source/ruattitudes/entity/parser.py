from arekit.common.news.entities_parser import BaseEntitiesParser
from arekit.contrib.source.ruattitudes.sentence.base import RuAttitudesSentence
from arekit.contrib.source.ruattitudes.text_object import TextObject


class RuAttitudesTextEntitiesParser(BaseEntitiesParser):

    def __init__(self):
        super(RuAttitudesTextEntitiesParser, self).__init__()
        self.__text_list = None
        self.__sentence = None

    def _before_parsing(self, sentence):
        assert(isinstance(sentence, RuAttitudesSentence))
        self.__text_list = sentence.get_text_as_list()
        self.__sentence = sentence

    def _iter_part(self, from_index, to_index):
        return self.__text_list[from_index:to_index]

    def _get_sentence_length(self):
        return len(self.__text_list)

    def _iter_subs_values_with_bounds(self):
        for text_object in self.__sentence.iter_objects():
            assert(isinstance(text_object, TextObject))
            entity = text_object.to_entity(lambda sent_id: self.__sentence.get_doc_level_text_object_id(sent_id))
            yield entity, text_object.Bound
