# -*- coding: utf-8 -*-
from core.processing.lemmatization.base import Stemmer
from core.processing.text.parser import ParsedText
from core.source.frames.variants import FrameVariantInText, FrameVariantsCollection


# TODO. This should be in source/frames/helper.py
class FramesHelper:

    def __init__(self, frames):
        assert(isinstance(frames, FrameVariantsCollection))
        self.__frames = frames

    def find_and_mark_frames(self, raw_terms, stemmer):
        assert(isinstance(raw_terms, list))
        assert(isinstance(stemmer, Stemmer))

        def __remove(terms, start, end):
            while end > start:
                del terms[start]
                end -= 1

        parsed_text = ParsedText(terms=raw_terms, hide_tokens=True, stemmer=stemmer)
        frame_variants = self.find_frames(parsed_text)

        if frame_variants is None:
            return raw_terms

        for variant in reversed(frame_variants):
            assert (isinstance(variant, FrameVariantInText))
            variant_bound = variant.get_bound()
            __remove(raw_terms,
                     start=variant_bound.TermIndex,
                     end=variant_bound.TermIndex + variant_bound.Length)
            raw_terms.insert(variant_bound.TermIndex, variant)

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

        def __check(terms, start_index, last_index):
            for i in range(start_index, last_index + 1):
                if not isinstance(terms[i], unicode):
                    return False
            return True

        def __check_inverted_frame_prefix(terms, index):
            if index == 0:
                return False
            if isinstance(terms[index - 1], unicode):
                return terms[index - 1] == u'не'
            return False

        assert(isinstance(parsed_text, ParsedText))

        text_frame_variants = []
        start_ind = 0
        last_ind = 0
        lemmas = list(parsed_text.iter_raw_lemmas())
        __replace_specific_russian_chars(lemmas)
        max_variant_len = max([len(variant) for _, variant in self.__frames.iter_variants()])
        while start_ind < len(lemmas):
            for ctx_size in reversed(range(1, max_variant_len)):

                last_ind = start_ind + ctx_size - 1

                if not(last_ind < len(lemmas)):
                    continue

                if not __check(lemmas, start_ind, last_ind):
                    continue

                ctx_template = u" ".join(lemmas[start_ind:last_ind + 1])

                if not self.__frames.has_variant(ctx_template):
                    continue

                frame_variant = FrameVariantInText(
                    variant=self.__frames.get_variant_by_template(ctx_template),
                    start_index=start_ind,
                    is_inverted=__check_inverted_frame_prefix(lemmas, start_ind))

                text_frame_variants.append(frame_variant)
                break

            start_ind = last_ind + 1

        if len(text_frame_variants) == 0:
            return None

        return text_frame_variants

