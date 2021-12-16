from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.languages.mods import BaseLanguageMods
from arekit.common.languages.ru.mods import RussianLanguageMods
from arekit.common.text.pipeline_ctx import PipelineContext
from arekit.common.text.pipeline_item import TextParserPipelineItem


class FrameVariantsParser(TextParserPipelineItem):

    def __init__(self, frame_variants, locale_mods=RussianLanguageMods):
        assert(isinstance(frame_variants, FrameVariantsCollection))
        assert(len(frame_variants) > 0)
        assert(issubclass(locale_mods, BaseLanguageMods))

        super(FrameVariantsParser, self).__init__()

        self.__frame_variants = frame_variants
        self.__max_variant_len = max([len(variant) for _, variant in frame_variants.iter_variants()])
        self._locale_mods = locale_mods

    # region private methods

    @staticmethod
    def __check_all_terms_within(terms, start_index, last_index):
        for term_ind in range(start_index, last_index + 1):
            if not isinstance(terms[term_ind], str):
                return False
        return True

    @staticmethod
    def __get_preposition(terms, index):
        return terms[index-1] if index > 0 else None

    def __try_compose_frame_variant(self, lemmas, start_ind, last_ind):

        if last_ind >= len(lemmas):
            return None

        is_all_words_within = self.__check_all_terms_within(
            terms=lemmas,
            start_index=start_ind,
            last_index=last_ind)

        if not is_all_words_within:
            return None

        ctx_value = " ".join(lemmas[start_ind:last_ind + 1])

        if not self.__frame_variants.has_variant(ctx_value):
            return None

        return ctx_value

    def _iter_processed(self, terms, origin):
        assert(len(terms) == len(origin))

        start_ind = 0
        last_ind = 0
        while start_ind < len(terms):

            found = False

            for ctx_size in reversed(list(range(1, self.__max_variant_len))):

                last_ind = start_ind + ctx_size - 1

                ctx_value = self.__try_compose_frame_variant(
                    start_ind=start_ind,
                    last_ind=last_ind,
                    lemmas=terms)

                if ctx_value is None:
                    continue

                prep_term = self.__get_preposition(terms=terms, index=start_ind)

                yield TextFrameVariant(
                    variant=self.__frame_variants.get_variant_by_value(ctx_value),
                    is_inverted=self._locale_mods.is_negation_word(prep_term) if prep_term is not None else False)

                found = True

                break

            if not found:
                yield origin[start_ind]

            start_ind = last_ind + 1

    # endregion

    def apply(self, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))

        # extract terms.
        terms = pipeline_ctx.provide("src")
        processed_it = self._iter_processed(terms=terms, origin=terms)

        # update the result.
        pipeline_ctx.update("src", value=list(processed_it))
