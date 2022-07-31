from arekit.common.frames.text_variant import TextFrameVariant
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.utils.processing.languages.mods import BaseLanguageMods
from arekit.contrib.utils.processing.languages.ru.mods import RussianLanguageMods


class FrameVariantsSentimentNegation(BasePipelineItem):

    def __init__(self, locale_mods=RussianLanguageMods):
        assert(issubclass(locale_mods, BaseLanguageMods))
        self._locale_mods = locale_mods

    @staticmethod
    def __get_preposition(terms, index):
        return terms[index-1] if index > 0 else None

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))
        assert(isinstance(pipeline_ctx, PipelineContext))

        for curr_ind, term in enumerate(input_data):

            if not isinstance(term, TextFrameVariant):
                continue

            prep_term = self.__get_preposition(terms=input_data, index=curr_ind)
            is_negated = self._locale_mods.is_negation_word(prep_term) if prep_term is not None else False

            term.set_is_negated(is_negated)

        return input_data
