from core.source.entity import Entity


class NewsHelper:

    def __init__(self, news):
        self.news = news
        self.__sentence_by_index_func = lambda index: news.get_sentence_by_index(index)
        self.__sentence_index_by_entity = NewsHelper.__index_sentence_by_entity(news)

    def get_sentence_by_entity(self, entity):
        assert(isinstance(entity, Entity))
        return self.__sentence_by_index_func(self.__sentence_index_by_entity[entity.ID])

    def get_sentence_index_by_entity(self, entity):
        assert(isinstance(entity, Entity))
        return self.__sentence_index_by_entity[entity.ID]

    @staticmethod
    def __index_sentence_by_entity(news):
        index = {}
        for sentence_index, sentence in enumerate(news.iter_sentences()):
            for e_ID in sentence.iter_entity_ids():
                index[e_ID] = sentence_index
        return index
