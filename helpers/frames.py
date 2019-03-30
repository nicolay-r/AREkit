# -*- coding: utf-8 -*-
from core.runtime.parser import ParsedText
from core.source.frames import FrameVariantInText, FramesCollection


class FramesHelper:

    def __init__(self, frames):
        assert(isinstance(frames, FramesCollection))
        self.__frames = frames

    def find_and_mark_frames(self, raw_terms):
        assert(isinstance(raw_terms, list))

        def __remove(terms, start, end):
            while end > start:
                del terms[start]
                end -= 1

        parsed_text = ParsedText(terms=raw_terms, hide_tokens=True)
        frame_variants = self.find_frames(parsed_text)

        if frame_variants is None:
            return raw_terms

        for variant in reversed(frame_variants):
            assert (isinstance(variant, FrameVariantInText))
            start, end = variant.get_bounds()
            __remove(raw_terms, start, end)
            raw_terms.insert(variant.start_index, variant)

        return raw_terms

    def find_frames(self, parsed_text):
        """
        Searching frames that a part of terms

        terms: ParsedText
            parsed text
        return: list or None
            list of tuples (frame, term_begin_index), or None
        """
        def __replace_specific_russian_chars(terms):
            for i, term in enumerate(terms):
                if not isinstance(term, unicode):
                    continue
                terms[i] = term.replace(u'ё', u'e')

        def __check(terms, start_ind, last_ind):
            for i in range(start_ind, last_ind + 1):
                if not isinstance(terms[i], unicode):
                    return False
            return True

        assert(isinstance(parsed_text, ParsedText))

        text_frame_variants = []
        start_ind = 0
        last_ind = 0
        terms = list(parsed_text.iter_raw_terms())
        __replace_specific_russian_chars(terms)
        max_variant_len = max([len(variant.terms) for _, variant in self.__frames.iter_variants()])
        while start_ind < len(terms):
            for ctx_size in reversed(range(1, max_variant_len)):

                last_ind = start_ind + ctx_size - 1

                if not(last_ind < len(terms)):
                    break

                if not __check(terms, start_ind, last_ind):
                    continue

                ctx_template = u" ".join(terms[start_ind:last_ind + 1])
                if self.__frames.has_variant(ctx_template):
                    frame_variant = FrameVariantInText(
                        self.__frames.get_variant_by_template(ctx_template),
                        start_ind)
                    text_frame_variants.append(frame_variant)
                    break

            start_ind = last_ind + 1

        if len(text_frame_variants) == 0:
            return None

        return text_frame_variants

