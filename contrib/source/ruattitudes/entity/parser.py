from arekit.common.news.entities_parser import BaseEntitiesParser
from arekit.contrib.source.ruattitudes.sentence.base import RuAttitudesSentence
from arekit.contrib.source.ruattitudes.text_object import TextObject


class RuAttitudesTextEntitiesParser(BaseEntitiesParser):

    def parse(self, sentence):
        assert(isinstance(sentence, RuAttitudesSentence))
        return self.__iter_terms_with_entities(sentence=sentence)

    @staticmethod
    def __iter_terms_with_entities(sentence):
        assert(isinstance(sentence, RuAttitudesSentence))
        subs_iter = RuAttitudesTextEntitiesParser.__iter_subs(sentence=sentence)
        return BaseEntitiesParser.iter_text_with_substitutions(text=sentence.get_text_as_list(),
                                                               iter_subs=subs_iter)

    @staticmethod
    def __iter_subs(sentence):
        assert(isinstance(sentence, RuAttitudesSentence))
        for text_object in sentence.iter_objects():
            assert(isinstance(text_object, TextObject))
            e = text_object.to_entity(to_doc_id_func=lambda sent_id: sentence.get_doc_level_text_object_id(sent_id))
            yield e, text_object.Bound

