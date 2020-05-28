from arekit.source.rusentrel.entities.entity import RuSentRelEntity


class RuSentRelNewsHelper:

    def __init__(self, news):
        self.news = news
        self.__sentence_index_by_entity = RuSentRelNewsHelper.__index_sentence_by_entity(news)

    # TODO. Duplicated in parsed text collection.
    # TODO. Duplicated in parsed text collection.
    # TODO. Duplicated in parsed text collection.
    def get_sentence_index_by_entity(self, entity):
        assert(isinstance(entity, RuSentRelEntity))
        return self.__sentence_index_by_entity[entity.IdInDocument]

    # region private methods

    # TODO. Duplicated in parsed text collection.
    # TODO. Duplicated in parsed text collection.
    # TODO. Duplicated in parsed text collection.
    @staticmethod
    def __index_sentence_by_entity(news):
        index = {}

        for sentence_index, sentence in enumerate(news.iter_sentences()):
            for e_ID in sentence.iter_entity_ids():
                assert(e_ID not in index)
                index[e_ID] = sentence_index

        return index

    # endregion
