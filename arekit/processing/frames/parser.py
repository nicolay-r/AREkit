from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.languages.ru.mods import RussianLanguageMods
from arekit.common.languages.mods import BaseLanguageMods


class FrameVariantsParser(object):

    # region private methods

    @staticmethod
    def __check_all_words_within(terms, start_index, last_index):
        for i in range(start_index, last_index + 1):
            if not isinstance(terms[i], str):
                return False
        return True

    @staticmethod
    def __get_preposition(terms, index):
        return terms[index-1] if index > 0 else None

    # endregion

    @staticmethod
    def iter_frames_from_lemmas(frame_variants, lemmas, locale_mods=RussianLanguageMods):
        """
        Considered to perform frames annotation across lemmatized terms.
        """
        assert(isinstance(frame_variants, FrameVariantsCollection))
        assert(isinstance(lemmas, list))
        assert(issubclass(locale_mods, BaseLanguageMods))
        assert(len(frame_variants) > 0)

        start_ind = 0
        last_ind = 0
        max_variant_len = max([len(variant) for _, variant in frame_variants.iter_variants()])
        while start_ind < len(lemmas):
            for ctx_size in reversed(list(range(1, max_variant_len))):

                last_ind = start_ind + ctx_size - 1

                if not(last_ind < len(lemmas)):
                    continue

                is_all_words_within = FrameVariantsParser.__check_all_words_within(
                    terms=lemmas,
                    start_index=start_ind,
                    last_index=last_ind)

                if not is_all_words_within:
                    continue

                ctx_value = " ".join(lemmas[start_ind:last_ind + 1])

                if not frame_variants.has_variant(ctx_value):
                    continue

                prep_term = FrameVariantsParser.__get_preposition(terms=lemmas,
                                                                  index=start_ind)

                yield TextFrameVariant(
                    variant=frame_variants.get_variant_by_value(ctx_value),
                    start_index=start_ind,
                    is_inverted=locale_mods.is_negation_word(prep_term) if prep_term is not None else False)

                break

            start_ind = last_ind + 1
