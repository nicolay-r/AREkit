from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news.base import News
from arekit.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNews(News):

    def __init__(self, sentences, news_index):
        assert(len(sentences) > 0)

        super(RuAttitudesNews, self).__init__(news_id=news_index,
                                              sentences=sentences,
                                              entities_parser=RuAttitudesTextEntitiesParser())

        self.__set_owners()
        self.__objects_before_sentence = self.__cache_objects_declared_before()

    # region properties

    @property
    def Title(self):
        return self._sentences[0]

    # endregion

    # region private methods

    def __set_owners(self):
        for sentence in self._sentences:
            assert(isinstance(sentence, RuAttitudesSentence))
            sentence.set_owner(self)

    def __cache_objects_declared_before(self):
        d = {}
        before = 0
        for s in self._sentences:
            assert(isinstance(s, RuAttitudesSentence))
            d[s.SentenceIndex] = before
            before += s.ObjectsCount

        return d

    # endregion

    # region public methods

    def get_sentence(self, index):
        return self._sentences[index]

    def get_objects_declared_before(self, sentence_index):
        return self.__objects_before_sentence[sentence_index]

    # endregion

    # region base News

    def iter_wrapped_linked_text_opinions(self, opinions):
        """
        Note: Complexity is O(N^2)
        """
        for opinion in opinions:
            yield LinkedTextOpinionsWrapper(self.__iter_all_text_opinions_in_sentences(opinion=opinion))

    # endregion

    # region Private methods

    def __iter_all_text_opinions_in_sentences(self, opinion):
        for sentence in self.iter_sentences(return_text=False):
            assert(isinstance(sentence, RuAttitudesSentence))

            ref_opinion = sentence.find_ref_opinion_by_key(key=opinion.Tag)
            if ref_opinion is None:
                continue

            yield ref_opinion.to_text_opinion(
                news_id=sentence.Owner.ID,
                end_to_doc_id_func=lambda sent_level_id: sentence.get_doc_level_text_object_id(sent_level_id),
                text_opinion_id=None)

    # endregion
