from arekit.common.context.terms_mapper import TextTermsMapper
from arekit.contrib.utils.processing.languages.pos import PartOfSpeechType
from arekit.contrib.utils.processing.pos.base import POSTagger


class PosTermsMapper(TextTermsMapper):

    def __init__(self, pos_tagger):
        assert(isinstance(pos_tagger, POSTagger))
        self.__pos_tagger = pos_tagger

    def map_word(self, w_ind, word):
        return self.__pos_tagger.get_term_pos(word)

    def map_token(self, t_ind, token):
        return PartOfSpeechType.Unknown

    def map_text_frame_variant(self, fv_ind, text_frame_variant):
        return self.__pos_tagger.get_term_pos(text_frame_variant.Variant.get_value())

    def map_entity(self, e_ind, entity):
        return PartOfSpeechType.Unknown
