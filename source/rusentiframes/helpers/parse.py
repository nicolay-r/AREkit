# -*- coding: utf-8 -*-
import collections

from core.common.frame_variants.collection import FrameVariantsCollection
from core.common.text_frame_variant import TextFrameVariant
from core.languages.mods import BaseLanguageMods
from core.languages.ru.mods import RussianLanguageMods
from core.processing.text.parsed import ParsedText
from core.source.rusentiframes.helpers.search import RuSentiFramesSearchHelper


class RuSentiFramesParseHelper:

    @staticmethod
    def parse_frames_in_parsed_text(frame_variants_collection, parsed_text, locale_mods=RussianLanguageMods):
        assert(isinstance(frame_variants_collection, FrameVariantsCollection))
        assert(isinstance(parsed_text, ParsedText))
        assert(issubclass(locale_mods, BaseLanguageMods))

        frame_variants_iter = RuSentiFramesSearchHelper.iter_frames_from_parsed_text(
            frame_variants=frame_variants_collection,
            parsed_text=parsed_text,
            locale_mods=locale_mods)

        if frame_variants_iter is None:
            return parsed_text

        updated_terms = RuSentiFramesParseHelper.__insert_frame_variants_into_raw_terms_list(
            raw_terms_list=list(parsed_text.iter_raw_terms()),
            frame_variants_iter=frame_variants_iter)

        return parsed_text.copy_modified(terms=updated_terms)

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
