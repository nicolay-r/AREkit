from core.processing.lemmatization.base import Stemmer
from core.runtime.parser import ParsedText
from core.source.frames.variants import FrameVariantInText, FrameVariantsCollection


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

        parsed_text = ParsedText(terms=raw_terms, stemmer=stemmer)
        frame_variants = self.find_frames(parsed_text.iter_lemmas(return_raw=True))

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

    def find_frames(self, lemmas):
        """
        Searching frames that a part of terms

        lemmas: list
            list of lemmatized terms
        return: list or None
            list of tuples (frame, term_begin_index), or None
        """
        def __replace_specific_russian_chars(terms):
            for i, term in enumerate(terms):
                if not isinstance(term, str):
                    continue
                terms[i] = term.replace('ё', 'e')

        def __check(terms, start_index, last_index):
            for i in range(start_index, last_index + 1):
                if not isinstance(terms[i], str):
                    return False
            return True

        def __check_inverted_frame_prefix(terms, index):
            if index == 0:
                return False
            if isinstance(terms[index - 1], str):
                return terms[index - 1] == 'не'
            return False

        assert(isinstance(lemmas, list))

        text_frame_variants = []
        start_ind = 0
        last_ind = 0
        __replace_specific_russian_chars(lemmas)
        max_variant_len = max([len(variant) for _, variant in self.__frames.iter_variants()])
        while start_ind < len(lemmas):
            for ctx_size in reversed(list(range(1, max_variant_len))):

                last_ind = start_ind + ctx_size - 1

                if not(last_ind < len(lemmas)):
                    continue

                if not __check(lemmas, start_ind, last_ind):
                    continue

                ctx_template = " ".join(lemmas[start_ind:last_ind + 1])

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

