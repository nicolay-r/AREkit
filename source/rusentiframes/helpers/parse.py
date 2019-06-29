# -*- coding: utf-8 -*-
from core.languages.mods import BaseLanguageMods
from core.languages.ru.mods import RussianLanguageMods
from core.processing.lemmatization.base import Stemmer
from core.processing.text.parsed import ParsedText
from core.source.rusentiframes.helpers.search import RuSentiFramesSearchHelper
from core.source.rusentiframes.variants.collection import FrameVariantsCollection
from core.source.rusentiframes.variants.text_variant import FrameVariantInText


class RuSentiFramesParseHelper:

    def __init__(self, frames):
        assert(isinstance(frames, FrameVariantsCollection))
        self.__frames = frames

    def parse_frames_in_raw_terms(self, raw_terms, stemmer, locale_mods=RussianLanguageMods):
        assert(isinstance(raw_terms, list))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(locale_mods, BaseLanguageMods))

        parsed_text = ParsedText(terms=raw_terms,
                                 hide_tokens=True,
                                 stemmer=stemmer)

        frame_variants = RuSentiFramesSearchHelper.iter_frames_from_parsed_text(
            frames=self.__frames,
            parsed_text=parsed_text,
            locale_mods=locale_mods)

        if frame_variants is None:
            return raw_terms

        return self.__insert_frame_variants_into_raw_terms_list(raw_terms=raw_terms,
                                                                frame_variants=frame_variants)

    def parse_frames_in_parsed_text(self, parsed_text):
        assert(isinstance(parsed_text, ParsedText))
        # TODO. Add method create_modified() in ParsedText.
        # TODO. Which allows to create new ParsedText with same settings.
        pass

    @staticmethod
    def __insert_frame_variants_into_raw_terms_list(raw_terms, frame_variants):
        assert(isinstance(raw_terms, list))
        assert(isinstance(frame_variants, list))

        def __remove(terms, start, end):
            while end > start:
                del terms[start]
                end -= 1

        for variant in reversed(frame_variants):
            assert (isinstance(variant, FrameVariantInText))
            variant_bound = variant.get_bound()
            __remove(terms=raw_terms,
                     start=variant_bound.Position,
                     end=variant_bound.Position + variant_bound.Length)
            raw_terms.insert(variant_bound.Position, variant)

        return raw_terms
