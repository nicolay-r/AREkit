from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentence import BratSentence
from arekit.contrib.source.ruattitudes.news import RuAttitudesNews
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion
from arekit.contrib.source.ruattitudes.opinions.converter import RuAttitudesSentenceOpinionConverter
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence
from arekit.common.utils import split_by_whitespaces


class RuAttitudesNewsConverter(object):
    """ Performs conversion to a brat-based representation.
        The latter allows then allows to adopt pipelines for TextOpnion extraction.
    """

    @staticmethod
    def to_brat_news(news):
        assert(isinstance(news, RuAttitudesNews))
        text_opinions = RuAttitudesNewsConverter.__iter_text_opinions(news=news)
        brat_sentences = RuAttitudesNewsConverter.__to_brat_sentences(news.iter_sentences())
        return BratNews(doc_id=news.ID,
                        sentences=brat_sentences,
                        text_relations=list(text_opinions))

    @staticmethod
    def __to_brat_sentences(sentences_iter):
        sentences = []
        for s in sentences_iter:
            assert(isinstance(s, RuAttitudesSentence))
            assert(s.Owner is not None)
            brat_entities = [obj.to_entity(s.get_doc_level_text_object_id) for obj in s.iter_objects()]
            brat_sentence = BratSentence(text=split_by_whitespaces(s.Text), index_begin=0, entities=brat_entities)
            sentences.append(brat_sentence)
        return sentences

    @staticmethod
    def __iter_text_opinions(news):
        assert(isinstance(news, RuAttitudesNews))
        for sentence in news.iter_sentences():
            assert(isinstance(sentence, RuAttitudesSentence))
            for sentence_opinion in sentence.iter_sentence_opins():
                assert(isinstance(sentence_opinion, SentenceOpinion))
                yield RuAttitudesSentenceOpinionConverter.to_brat_relation(
                    sentence_opinion=sentence_opinion,
                    end_to_doc_id_func=sentence.get_doc_level_text_object_id)
