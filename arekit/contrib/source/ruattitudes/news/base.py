from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.base import News
from arekit.common.opinions.base import Opinion
from arekit.contrib.source.ruattitudes.news.opin_converter import RuAttitudesSentenceOpinionConverter
from arekit.contrib.source.ruattitudes.sentence.base import RuAttitudesSentence


class RuAttitudesNews(News):

    def __init__(self, sentences, news_index):
        assert(len(sentences) > 0)

        super(RuAttitudesNews, self).__init__(doc_id=news_index, sentences=sentences)

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

    def get_objects_declared_before(self, sentence_index):
        return self.__objects_before_sentence[sentence_index]

    # endregion

    # region base News

    def extract_text_opinions_linkages(self, opinion, label_scaler):
        """
        Note: Complexity is O(N)
        """
        assert(isinstance(opinion, Opinion))
        assert(isinstance(label_scaler, BaseLabelScaler))

        return TextOpinionsLinkage(self.__iter_all_text_opinions_in_sentences(
            opinion=opinion, label_scaler=label_scaler))

    # endregion

    # region Private methods

    def __iter_all_text_opinions_in_sentences(self, opinion, label_scaler):
        for sentence in self.iter_sentences():
            assert(isinstance(sentence, RuAttitudesSentence))

            sentence_opin = sentence.find_sentence_opin_by_key(key=opinion.Tag)
            if sentence_opin is None:
                continue

            yield RuAttitudesSentenceOpinionConverter.to_text_opinion(
                sentence_opinion=sentence_opin,
                doc_id=sentence.Owner.ID,
                end_to_doc_id_func=lambda sent_level_id: sentence.get_doc_level_text_object_id(sent_level_id),
                text_opinion_id=None,
                label_scaler=label_scaler)

    # endregion
