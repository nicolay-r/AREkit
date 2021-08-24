from arekit.common.news.objects_parser import BaseObjectsParser
from arekit.contrib.source.ruattitudes.text_object import TextObject


class RuAttitudesTextEntitiesParser(BaseObjectsParser):

    def __init__(self):
        super(RuAttitudesTextEntitiesParser, self).__init__(
            iter_objs_func=self.__iter_subs_values_with_bounds)

    @staticmethod
    def __iter_subs_values_with_bounds(sentence):
        for text_object in sentence.iter_objects():
            assert(isinstance(text_object, TextObject))
            entity = text_object.to_entity(lambda sent_id: sentence.get_doc_level_text_object_id(sent_id))
            yield entity, text_object.Bound
