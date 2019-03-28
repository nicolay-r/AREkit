# -*- coding: utf-8 -*-
from core.runtime.parser import ParsedText
from core.source.frames import FrameVariantInText, FramesCollection


class FramesHelper:

    def __init__(self, frames):
        assert(isinstance(frames, FramesCollection))
        self.__frames = frames

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
                terms[i] = term.replace(u'ё', u'e')

        assert(isinstance(parsed_text, ParsedText))

        result = []
        start_ind = 0
        last_ind = 0
        terms = list(parsed_text.Terms)
        __replace_specific_russian_chars(terms)
        max_variant_len = max([len(variant.terms) for _, variant in self.__frames.iter_variants()])
        while start_ind < len(terms):
            for ctx_size in reversed(range(1, max_variant_len)):

                last_ind = start_ind + ctx_size - 1

                if not(last_ind < len(terms)):
                    break

                ctx_template = u" ".join(terms[start_ind:last_ind + 1])
                if self.__frames.has_variant(ctx_template):
                    result.append(FrameVariantInText(self.__frames.get_variant_by_template(ctx_template), start_ind))
                    break

            start_ind = last_ind + 1

        if len(result) == 0:
            return None

        return result

