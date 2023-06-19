from arekit.contrib.source.brat.doc import BratDocument
from arekit.contrib.source.brat.sentence import BratSentence
from arekit.contrib.source.ruattitudes.doc import RuAttitudesDocument
from arekit.contrib.source.ruattitudes.opinions.base import SentenceOpinion
from arekit.contrib.source.ruattitudes.opinions.converter import RuAttitudesSentenceOpinionConverter
from arekit.contrib.source.ruattitudes.sentence import RuAttitudesSentence
from arekit.common.utils import split_by_whitespaces


class RuAttitudesDocumentsConverter(object):
    """ Performs conversion to a brat-based representation.
        The latter allows then allows to adopt pipelines for TextOpnion extraction.
    """

    @staticmethod
    def to_brat_doc(doc):
        assert(isinstance(doc, RuAttitudesDocument))
        text_opinions = RuAttitudesDocumentsConverter.__iter_text_opinions(doc=doc)
        brat_sentences = RuAttitudesDocumentsConverter.__to_brat_sentences(doc.iter_sentences())
        return BratDocument(doc_id=doc.ID,
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
    def __iter_text_opinions(doc):
        assert(isinstance(doc, RuAttitudesDocument))
        for sentence in doc.iter_sentences():
            assert(isinstance(sentence, RuAttitudesSentence))
            for sentence_opinion in sentence.iter_sentence_opins():
                assert(isinstance(sentence_opinion, SentenceOpinion))
                yield RuAttitudesSentenceOpinionConverter.to_brat_relation(
                    sentence_opinion=sentence_opinion,
                    end_to_doc_id_func=sentence.get_doc_level_text_object_id)
