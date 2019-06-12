import collections

from core.common.object import TextObject
from core.common.ref_opinon import RefOpinion
from core.source.rusentrel.opinions.opinion import Opinion
from core.source.ruattitudes.news import ProcessedNews
from core.source.ruattitudes.sentence import ProcessedSentence


class NewsProcessingHelper(object):

    @staticmethod
    def build_opinion_dict(processed_news):
        return NewsProcessingHelper.__build_opinion_dict(processed_news)

    @staticmethod
    def to_processed_news_dict(processed_sentence_list):
        assert(isinstance(processed_sentence_list, collections.Iterable))
        docs = {}

        for s in processed_sentence_list:
            assert(isinstance(s, ProcessedSentence))
            assert(isinstance(s.Owner, ProcessedNews))
            news_id = s.Owner.NewsIndex

            if news_id in docs:
                continue

            docs[news_id] = s.Owner

        return docs

    @staticmethod
    def iter_opinions_with_related_sentences(processed_news):
        assert(isinstance(processed_news, ProcessedNews))

        doc_opinions = NewsProcessingHelper.build_opinion_dict(processed_news=processed_news)
        assert(isinstance(doc_opinions, dict))

        for ref_opinion_tag, value in doc_opinions.iteritems():

            opinion = None
            related_sentences = []

            for sentence in processed_news.iter_processed_sentences():

                ref_opinion = sentence.find_ref_opinion_by_key(ref_opinion_tag)
                if ref_opinion is None:
                    continue

                related_sentences.append(sentence)

                if opinion is not None:
                    continue

                opinion = NewsProcessingHelper.__convert_ref_opinion_to_opinion(sentence=sentence,
                                                                                ref_opinion=ref_opinion)

            if len(related_sentences) == 0:
                continue

            yield opinion, related_sentences

    @staticmethod
    def __convert_ref_opinion_to_opinion(sentence, ref_opinion):
        assert(isinstance(sentence, ProcessedSentence))
        assert(isinstance(ref_opinion, RefOpinion))

        l_obj, r_obj = sentence.get_objects(ref_opinion)

        assert (isinstance(l_obj, TextObject))
        assert (isinstance(r_obj, TextObject))

        return Opinion(value_left=l_obj.get_value(),
                       value_right=r_obj.get_value(),
                       sentiment=ref_opinion.Sentiment)

    @staticmethod
    def __build_opinion_dict(processed_news):
        opin_dict = {}

        for s_ind, sentence in enumerate(processed_news.iter_processed_sentences()):
            assert(isinstance(sentence, ProcessedSentence))
            for ref_opinion in sentence.iter_ref_opinions():
                assert(isinstance(ref_opinion, RefOpinion))
                key = ref_opinion.Tag
                if key not in opin_dict:
                    opin_dict[key] = []
                opin_dict[key].append(s_ind)

        return opin_dict
