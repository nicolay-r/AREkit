import collections

from core.common.text_object import TextObject
from core.common.ref_opinon import RefOpinion
from core.common.opinions.base import Opinion
from core.source.ruattitudes.news import RuAttitudesNews
from core.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesNewsHelper(object):

    # region public methods

    @staticmethod
    def build_opinion_dict(news):
        return RuAttitudesNewsHelper.__build_opinion_dict(news)

    @staticmethod
    def to_news_dict(sentence_list):
        assert(isinstance(sentence_list, collections.Iterable))
        docs = {}

        for s in sentence_list:
            assert(isinstance(s, RuAttitudesSentence))
            assert(isinstance(s.Owner, RuAttitudesNews))
            news_id = s.Owner.NewsIndex

            if news_id in docs:
                continue

            docs[news_id] = s.Owner

        return docs

    @staticmethod
    def iter_opinions_with_related_sentences(news):
        assert(isinstance(news, RuAttitudesNews))

        doc_opinions = RuAttitudesNewsHelper.build_opinion_dict(news=news)
        assert(isinstance(doc_opinions, dict))

        for ref_opinion_tag, value in doc_opinions.iteritems():

            opinion, related_sentences = RuAttitudesNewsHelper.__extract_opinion_with_related_sentences(
                news=news,
                ref_opinion_tag=ref_opinion_tag)

            if opinion is None:
                continue

            yield opinion, related_sentences

    # endregion

    # region private methods

    @staticmethod
    def __extract_opinion_with_related_sentences(news, ref_opinion_tag):
        opinion = None
        related_sentences = []

        for sentence in news.iter_sentences():

            ref_opinion = sentence.find_ref_opinion_by_key(ref_opinion_tag)
            if ref_opinion is None:
                continue

            related_sentences.append(sentence)

            if opinion is not None:
                continue

            opinion = RuAttitudesNewsHelper.__convert_ref_opinion_to_opinion(sentence=sentence,
                                                                             ref_opinion=ref_opinion)

        return opinion, related_sentences

    @staticmethod
    def __convert_ref_opinion_to_opinion(sentence, ref_opinion):
        assert(isinstance(sentence, RuAttitudesSentence))
        assert(isinstance(ref_opinion, RefOpinion))

        l_obj, r_obj = sentence.get_objects(ref_opinion)

        assert (isinstance(l_obj, TextObject))
        assert (isinstance(r_obj, TextObject))

        opinion = Opinion(source_value=l_obj.get_value(),
                          target_value=r_obj.get_value(),
                          sentiment=ref_opinion.Sentiment)

        # Using this tag allows to perform a revert operation,
        # i.e. to find opinion_ref by opinion.
        opinion.set_tag(ref_opinion.Tag)

        return opinion

    @staticmethod
    def __build_opinion_dict(news):
        opin_dict = {}

        for s_ind, sentence in enumerate(news.iter_sentences()):
            assert(isinstance(sentence, RuAttitudesSentence))
            for ref_opinion in sentence.iter_ref_opinions():
                assert(isinstance(ref_opinion, RefOpinion))
                key = ref_opinion.Tag
                if key not in opin_dict:
                    opin_dict[key] = []
                opin_dict[key].append(s_ind)

        return opin_dict

    # endregion
