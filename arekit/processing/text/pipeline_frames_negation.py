from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.item import BasePipelineItem
from arekit.processing.languages.mods import BaseLanguageMods
from arekit.processing.languages.ru.mods import RussianLanguageMods


class FrameVariantsSentimentNegation(BasePipelineItem):

    def __init__(self, locale_mods=RussianLanguageMods):
        assert(issubclass(locale_mods, BaseLanguageMods))
        self._locale_mods = locale_mods

    @staticmethod
    def __get_preposition(terms, index):
        return terms[index-1] if index > 0 else None

    def __update(self, terms):

        for curr_ind, term in enumerate(terms):

            if not isinstance(term, TextFrameVariant):
                continue

            prep_term = self.__get_preposition(terms=terms, index=curr_ind)
            is_negated = self._locale_mods.is_negation_word(prep_term) if prep_term is not None else False

            term.set_is_negated(is_negated)

    def apply(self, pipeline_ctx):
        assert (isinstance(pipeline_ctx, PipelineContext))

        # extract terms.
        terms_list = list(pipeline_ctx.provide("src"))
        self.__update(terms_list)

        # update the result.
        pipeline_ctx.update("src", value=terms_list)
