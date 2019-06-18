from core.source.rusentrel.entities.entity import RuSentRelEntity


class NewsHelper:

    def __init__(self, news):
        self.news = news
        self.__sentence_by_index_func = lambda index: news.get_sentence_by_index(index)
        self.__sentence_index_by_entity = NewsHelper.__index_sentence_by_entity(news)

    def get_sentence_by_entity(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        return self.__sentence_by_index_func(self.__sentence_index_by_entity[entity.IdInDocument])

    def get_sentence_index_by_entity(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        return self.__sentence_index_by_entity[entity.IdInDocument]

    @staticmethod
    def __index_sentence_by_entity(news):
        index = {}
        for sentence_index, sentence in enumerate(news.iter_sentences()):
            for e_ID in sentence.iter_entity_ids():
                index[e_ID] = sentence_index
        return index
