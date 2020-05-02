from arekit.common.linked_text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.news import News
from arekit.common.text_opinions.base import RefOpinion
from arekit.common.text_opinions.text_opinion import TextOpinion
from arekit.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNews(News):

    def __init__(self, sentences, news_index):
        assert(isinstance(sentences, list))
        assert(len(sentences) > 0)
        assert(isinstance(news_index, int))

        super(News, self).__init__()

        self.__sentences = sentences
        self.__news_index = news_index
        self.__set_owners()
        self.__objects_before_sentence = self.__cache_objects_declared_before()

    # region properties

    @property
    def Title(self):
        return self.__sentences[0]

    @property
    def NewsIndex(self):
        return self.__news_index

    # endregion

    # region private methods

    def __set_owners(self):
        for sentence in self.__sentences:
            assert(isinstance(sentence, RuAttitudesSentence))
            sentence.set_owner(self)

    def __cache_objects_declared_before(self):
        d = {}
        before = 0
        for s in self.__sentences:
            assert(isinstance(s, RuAttitudesSentence))
            d[s.SentenceIndex] = before
            before += s.ObjectsCount

        return d

    # endregion

    # region public methods

    def get_sentence(self, index):
        return self.__sentences[index]

    def get_objects_declared_before(self, sentence_index):
        return self.__objects_before_sentence[sentence_index]

    def iter_sentences(self):
        for sentence in self.__sentences:
            yield sentence

    # endregion

    # region base News

    def iter_wrapped_linked_text_opinions(self, opinions):
        """
        Note: Complexity is O(N^2)
        """
        for opinion in opinions:
            yield LinkedTextOpinionsWrapper(self.__iter_all_text_opinions_in_sentences(opinion=opinion))

    # region Private methods

    def __iter_all_text_opinions_in_sentences(self, opinion):
        for sentence in self.iter_sentences():
            assert(isinstance(sentence, RuAttitudesSentence))

            ref_opinion = sentence.find_ref_opinion_by_key(key=opinion.Tag)

            if ref_opinion is None:
                continue

            yield self.__ref_opinion_to_text_opinion(
                news_index=sentence.Owner.NewsIndex,
                ref_opinion=ref_opinion,
                sent_to_doc_id_func=sentence.get_doc_level_text_object_id)

    @staticmethod
    def __ref_opinion_to_text_opinion(news_index,
                                      ref_opinion,
                                      sent_to_doc_id_func):
        assert(isinstance(news_index, int))
        assert(isinstance(ref_opinion, RefOpinion))
        assert(callable(sent_to_doc_id_func))

        cloned_ref_opinion = RefOpinion(
            source_id=sent_to_doc_id_func(ref_opinion.SourceId),
            target_id=sent_to_doc_id_func(ref_opinion.TargetId),
            sentiment=ref_opinion.Sentiment)

        return TextOpinion.create_from_ref_opinion(
            news_id=news_index,
            text_opinion_id=None,
            ref_opinion=cloned_ref_opinion)

    # endregion

    # endregion
