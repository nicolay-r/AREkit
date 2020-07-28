import collections

from arekit.common.entities.base import Entity
from arekit.common.news.entities_parser import BaseEntitiesParser
from arekit.source.ruattitudes.text_object import TextObject
from arekit.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesTextEntitiesParser(BaseEntitiesParser):

    def __init__(self, iter_sentences):
        """
        objects_in_sent: list of pairs (int, int)
            list of (sentence_id, amount of objects in every news sentence).
        """
        assert(isinstance(iter_sentences, collections.Iterable))

        self.__objs_count_per_sent = {}

        objs_before = 0
        for s in iter_sentences:
            assert(isinstance(s, RuAttitudesSentence))
            self.__objs_count_per_sent[s.SentenceIndex] = s.ObjectsCount + objs_before
            objs_before += s.ObjectsCount

    def parse(self, sentence):
        assert(isinstance(sentence, RuAttitudesSentence))

        return self.__iter_terms_with_entities(
            sentence=sentence,
            s_to_doc_id=lambda s_level_id: s_level_id + self.__objs_count_per_sent[sentence.SentenceIndex])

    @staticmethod
    def __iter_terms_with_entities(sentence, s_to_doc_id):
        assert(isinstance(sentence, RuAttitudesSentence))
        assert(callable(s_to_doc_id))

        subs_iter = RuAttitudesTextEntitiesParser.__iter_subs(sentence=sentence,
                                                              s_to_doc_id=s_to_doc_id)

        return BaseEntitiesParser.iter_text_with_substitutions(text=sentence.Text,
                                                               iter_subs=subs_iter)

    @staticmethod
    def __iter_subs(sentence, s_to_doc_id):
        assert(isinstance(sentence, RuAttitudesSentence))
        assert(callable(s_to_doc_id))

        for s_level_id, obj in enumerate(sentence.iter_objects()):
            assert(isinstance(obj, TextObject))

            _value = obj.get_value()
            value = _value if len(_value) > 0 else u'[empty]'

            entity = Entity(value=value,
                            e_type=obj.Type,
                            id_in_doc=s_to_doc_id(s_level_id))

            yield entity, obj.get_bound()

