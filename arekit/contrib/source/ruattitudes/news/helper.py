from arekit.contrib.source.ruattitudes.news.base import RuAttitudesNews
from arekit.contrib.source.ruattitudes.sentence.base import RuAttitudesSentence
from arekit.contrib.source.ruattitudes.sentence.opinion import SentenceOpinion


class RuAttitudesNewsHelper(object):

    # region public methods

    @staticmethod
    def build_opinion_dict(news):
        return RuAttitudesNewsHelper.__build_opinion_dict(news)

    @staticmethod
    def iter_opinions_with_related_sentences(news):
        assert(isinstance(news, RuAttitudesNews))

        doc_opinions = RuAttitudesNewsHelper.build_opinion_dict(news=news)
        assert(isinstance(doc_opinions, dict))

        for sentence_opin_tag, value in doc_opinions.iteritems():

            opinion, related_sentences = RuAttitudesNewsHelper.__extract_opinion_with_related_sentences(
                news=news,
                sentence_opin_tag=sentence_opin_tag)

            if opinion is None:
                continue

            yield opinion, related_sentences

    # endregion

    # region private methods

    @staticmethod
    def __extract_opinion_with_related_sentences(news, sentence_opin_tag):
        opinion = None
        related_sentences = []

        for sentence in news.iter_sentences(return_text=False):
            assert(isinstance(sentence, RuAttitudesSentence))

            sentence_opin = sentence.find_sentence_opin_by_key(sentence_opin_tag)
            if sentence_opin is None:
                continue

            assert(isinstance(sentence_opin, SentenceOpinion))

            related_sentences.append(sentence)

            if opinion is not None:
                continue

            opinion = sentence_opin.to_opinion()

        return opinion, related_sentences

    @staticmethod
    def __build_opinion_dict(news):
        opin_dict = {}

        for s_ind, sentence in enumerate(news.iter_sentences(return_text=False)):
            assert(isinstance(sentence, RuAttitudesSentence))
            for sentence_opin in sentence.iter_sentence_opins():
                assert(isinstance(sentence_opin, SentenceOpinion))
                key = sentence_opin.Tag
                if key not in opin_dict:
                    opin_dict[key] = []
                opin_dict[key].append(s_ind)

        return opin_dict

    # endregion
