from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.source.ruattitudes.news import RuAttitudesDocument
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion
from arekit.contrib.source.ruattitudes.opinions.converter import RuAttitudesSentenceOpinionConverter
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence


class RuAttitudesSentenceOpinionUtils(object):

    # region public methods

    @staticmethod
    def iter_opinions_with_related_sentences(news, label_scaler):
        """ Provides opinions with the related sentences.
        """
        assert(isinstance(news, RuAttitudesDocument))
        assert(isinstance(label_scaler, BaseLabelScaler))

        doc_opinions = RuAttitudesSentenceOpinionUtils.__build_opinion_dict(news=news)
        assert(isinstance(doc_opinions, dict))

        for sentence_opin_tag, value in doc_opinions.items():

            opinion, related_sentences = RuAttitudesSentenceOpinionUtils.__extract_opinion_with_related_sentences(
                news=news, sentence_opin_tag=sentence_opin_tag, label_scaler=label_scaler)

            if opinion is None:
                continue

            yield opinion, related_sentences

    # endregion

    # region private methods

    @staticmethod
    def __extract_opinion_with_related_sentences(news, sentence_opin_tag, label_scaler):
        opinion = None
        related_sentences = []

        for sentence in news.iter_sentences():
            assert(isinstance(sentence, RuAttitudesSentence))

            sentence_opin = sentence.find_sentence_opin_by_key(sentence_opin_tag)
            if sentence_opin is None:
                continue

            assert(isinstance(sentence_opin, SentenceOpinion))

            related_sentences.append(sentence)

            if opinion is not None:
                continue

            source, target = sentence.get_objects(sentence_opin)

            opinion = RuAttitudesSentenceOpinionConverter.to_opinion(
                sentence_opinion=sentence_opin,
                source_value=source.Value,
                target_value=target.Value,
                label_scaler=label_scaler)

        return opinion, related_sentences

    @staticmethod
    def __build_opinion_dict(news):
        opin_dict = {}

        for s_ind, sentence in enumerate(news.iter_sentences()):
            assert(isinstance(sentence, RuAttitudesSentence))
            for sentence_opin in sentence.iter_sentence_opins():
                assert(isinstance(sentence_opin, SentenceOpinion))
                key = sentence_opin.Tag
                if key not in opin_dict:
                    opin_dict[key] = []
                opin_dict[key].append(s_ind)

        return opin_dict

    # endregion
