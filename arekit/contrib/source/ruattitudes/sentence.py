from arekit.common.news.sentence import BaseNewsSentence
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion


class RuAttitudesSentence(BaseNewsSentence):

    def __init__(self, is_title, text, sentence_opins, objects_list, sentence_index):
        assert(isinstance(is_title, bool))
        assert(isinstance(sentence_opins, list))
        assert(isinstance(objects_list, list))
        assert(isinstance(sentence_index, int))
        super(RuAttitudesSentence, self).__init__(text)

        self.__is_title = is_title
        self.__sentence_opins = sentence_opins
        self.__objects = objects_list
        self.__sentence_index = sentence_index
        self.__owner = None

    # region properties

    @property
    def SentenceIndex(self):
        return self.__sentence_index

    @property
    def IsTitle(self):
        return self.__is_title

    @property
    def Owner(self):
        return self.__owner

    @property
    def ObjectsCount(self):
        return len(self.__objects)

    # endregion

    # region public methods

    def set_owner(self, owner):
        if self.__owner is not None:
            raise Exception("Owner is already declared")
        self.__owner = owner

    def get_objects(self, sentence_opin):
        assert(isinstance(sentence_opin, SentenceOpinion))
        source_obj = self.__objects[sentence_opin.SourceID]
        target_obj = self.__objects[sentence_opin.TargetID]
        return source_obj, target_obj

    def get_doc_level_text_object_id(self, text_object_ind):
        return text_object_ind + self.__owner.get_objects_declared_before(self.SentenceIndex)

    def iter_objects(self):
        for object in self.__objects:
            yield object

    def find_sentence_opin_by_key(self, key):
        assert(key is not None)

        for opinion in self.__sentence_opins:
            if opinion.Tag == key:
                return opinion

        return None

    def iter_sentence_opins(self):
        for opinion in self.__sentence_opins:
            yield opinion

    # endregion
