from core.languages.mods import BaseLanguageMods
from core.languages.ru.mods import RussianLanguageMods
from core.processing.text.parsed import ParsedText
from core.source.rusentiframes.variants.collection import FrameVariantsCollection
from core.source.rusentiframes.variants.text_variant import FrameVariantInText


class RuSentiFramesSearchHelper(object):

    @staticmethod
    def iter_frames_from_parsed_text(frames, parsed_text, locale_mods=RussianLanguageMods):
        assert(isinstance(frames, FrameVariantsCollection))
        assert(isinstance(parsed_text, ParsedText))
        assert(isinstance(locale_mods, BaseLanguageMods))

        lemmas_iter = parsed_text.iter_raw_lemmas()
        lemmas = [locale_mods.replace_specific_word_chars(lemma)
                  for lemma in lemmas_iter]

        start_ind = 0
        last_ind = 0
        max_variant_len = max([len(variant) for _, variant in frames.iter_variants()])
        while start_ind < len(lemmas):
            for ctx_size in reversed(range(1, max_variant_len)):

                last_ind = start_ind + ctx_size - 1

                if not(last_ind < len(lemmas)):
                    continue

                is_all_words_within = RuSentiFramesSearchHelper.__check_all_words_within(
                    terms=lemmas,
                    start_index=start_ind,
                    last_index=last_ind)

                if not is_all_words_within:
                    continue

                ctx_template = u" ".join(lemmas[start_ind:last_ind + 1])

                if not frames.has_variant(ctx_template):
                    continue

                prep_term = RuSentiFramesSearchHelper.__get_preposition(terms=lemmas,
                                                                        index=start_ind)

                yield FrameVariantInText(
                    variant=frames.get_variant_by_template(ctx_template),
                    start_index=start_ind,
                    is_inverted=locale_mods.is_negation_word(prep_term) if prep_term is not None else False)

                break

            start_ind = last_ind + 1

    @staticmethod
    def __check_all_words_within(terms, start_index, last_index):
        for i in range(start_index, last_index + 1):
            if not isinstance(terms[i], unicode):
                return False
        return True

    @staticmethod
    def __get_preposition(terms, index):
        return terms[index-1] if index > 0 else None
