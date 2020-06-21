from arekit.common.sentence import BaseNewsSentence
from arekit.common.text_opinions.base import RefOpinion


class RuAttitudesSentence(BaseNewsSentence):

    def __init__(self, is_title, text, ref_opinions, objects_list, sentence_index):
        assert(isinstance(is_title, bool))
        assert(isinstance(text, unicode))
        assert(isinstance(ref_opinions, list))
        assert(isinstance(objects_list, list))
        assert(isinstance(sentence_index, int))

        super(RuAttitudesSentence, self).__init__(text=text)

        self.__is_title = is_title
        self.__ref_opinions = ref_opinions
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

    def get_objects(self, ref_opinion):
        assert(isinstance(ref_opinion, RefOpinion))
        source_obj = self.__objects[ref_opinion.SourceId]
        target_obj = self.__objects[ref_opinion.TargetId]
        return source_obj, target_obj

    def get_doc_level_text_object_id(self, text_object_ind):
        return text_object_ind + \
               self.__owner.get_objects_declared_before(self.SentenceIndex)

    def iter_objects(self):
        for object in self.__objects:
            yield object

    def find_ref_opinion_by_key(self, key):
        assert(key is not None)

        for opinion in self.__ref_opinions:
            if opinion.Tag == key:
                return opinion

        return None

    def iter_ref_opinions(self):
        for opinion in self.__ref_opinions:
            yield opinion

    def __len__(self):
        return len(self.Text)

    # endregion
