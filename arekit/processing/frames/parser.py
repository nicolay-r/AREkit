import collections

from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.languages.ru.mods import RussianLanguageMods
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.common.languages.mods import BaseLanguageMods
from arekit.processing.frames.annot import FrameVariantsAnnotationHelper
from arekit.processing.text.enums import TermFormat
from arekit.processing.text.parsed import ParsedText


class FrameVariantsParser(object):

    # region private methods

    @staticmethod
    def __insert_frame_variants_into_raw_terms_list(raw_terms_list, frame_variants_iter):
        assert(isinstance(raw_terms_list, list))
        assert(isinstance(frame_variants_iter, collections.Iterable))

        def __remove(terms, start, end):
            while end > start:
                del terms[start]
                end -= 1

        for variant in reversed(list(frame_variants_iter)):
            assert (isinstance(variant, TextFrameVariant))
            variant_bound = variant.get_bound()
            __remove(terms=raw_terms_list,
                     start=variant_bound.Position,
                     end=variant_bound.Position + variant_bound.Length)
            raw_terms_list.insert(variant_bound.Position, variant)

        return raw_terms_list

    # endregion

    @staticmethod
    def parse_frames_in_parsed_text(frame_variants_collection, parsed_text, locale_mods=RussianLanguageMods):
        assert(isinstance(frame_variants_collection, FrameVariantsCollection))
        assert(isinstance(parsed_text, ParsedText))
        assert(issubclass(locale_mods, BaseLanguageMods))

        lemmas = [locale_mods.replace_specific_word_chars(lemma) if isinstance(lemma, str) else lemma
                  for lemma in parsed_text.iter_terms(term_format=TermFormat.Lemma)]

        frame_variants_iter = FrameVariantsAnnotationHelper.iter_frames_from_lemmas(
            frame_variants=frame_variants_collection,
            lemmas=lemmas,
            locale_mods=locale_mods)

        if frame_variants_iter is None:
            return parsed_text

        updated_terms = FrameVariantsParser.__insert_frame_variants_into_raw_terms_list(
            raw_terms_list=list(parsed_text.iter_terms(TermFormat.Raw)),
            frame_variants_iter=frame_variants_iter)

        return parsed_text.copy_modified(terms=updated_terms)
