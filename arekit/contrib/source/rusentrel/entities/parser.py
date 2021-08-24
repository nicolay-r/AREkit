from arekit.common.news.objects_parser import BaseObjectsParser


class RuSentRelTextEntitiesParser(BaseObjectsParser):

    def __init__(self):
        super(RuSentRelTextEntitiesParser, self).__init__(
            iter_objs_func=self.__iter_subs_values_with_bounds)

    @staticmethod
    def __iter_subs_values_with_bounds(sentence):
        return sentence.iter_entity_with_local_bounds()
