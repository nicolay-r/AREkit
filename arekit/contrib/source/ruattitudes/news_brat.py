from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.ruattitudes.news import RuAttitudesNews
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion
from arekit.contrib.source.ruattitudes.opinions.converter import RuAttitudesSentenceOpinionConverter
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNewsConverter(object):
    """ Performs conversion to a brat-based representation.
        The latter allows then allows to adopt pipelines for TextOpnion extraction.
    """

    @staticmethod
    def to_brat_news(news, label_scaler):
        assert(isinstance(news, RuAttitudesNews))
        assert(isinstance(label_scaler, BaseLabelScaler))
        text_opinions = RuAttitudesNewsConverter.__iter_text_opinions(news=news, label_scaler=label_scaler)
        return BratNews(doc_id=news.ID,
                        sentences=list(news.iter_sentences()),
                        text_opinions=list(text_opinions))

    @staticmethod
    def __iter_text_opinions(news, label_scaler):
        assert(isinstance(news, RuAttitudesNews))
        for sentence in news.iter_sentences():
            assert(isinstance(sentence, RuAttitudesSentence))
            for sentence_opinion in sentence.iter_sentence_opins():
                assert(isinstance(sentence_opinion, SentenceOpinion))
                yield RuAttitudesSentenceOpinionConverter.to_text_opinion(
                    doc_id=news.ID,
                    sentence_opinion=sentence_opinion,
                    end_to_doc_id_func=sentence.get_doc_level_text_object_id,
                    text_opinion_id=sentence_opinion.ID,
                    label_scaler=label_scaler)
