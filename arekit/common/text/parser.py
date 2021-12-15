from arekit.common.frames.variants.collection import FrameVariantsCollection

from arekit.common.languages.mods import BaseLanguageMods
from arekit.common.languages.ru.mods import RussianLanguageMods
from arekit.common.text.options import TextParseOptions
from arekit.common.text.parsed import BaseParsedText
from arekit.common.text.pipeline_ctx import PipelineContext
from arekit.common.text.pipeline_item import TextParserPipelineItem
from arekit.processing.frames.parser import FrameVariantsParser
from arekit.processing.text.enums import TermFormat


class BaseTextParser(object):

    def __init__(self, parse_options, pipeline):
        assert(isinstance(parse_options, TextParseOptions))
        assert(isinstance(pipeline, list))
        self._parse_options = parse_options
        self.__pipeline = pipeline

    def parse(self, pipeline_ctx):
        """
        terms_list: list of terms
        returns:
            ParsedText
        """
        assert(isinstance(pipeline_ctx, PipelineContext))

        for item in self.__pipeline:
            assert(isinstance(item, TextParserPipelineItem))
            item.apply(pipeline_ctx)

        # compose parsed text.
        parsed_text = BaseParsedText(terms=pipeline_ctx.provide("src"))

        # TODO. In further, this is considered to be departed from base text parser
        # TODO. and treated as an (optional) element of the text processing pipeline.
        # Frames parsing stage. (PPL 2).
        if self._parse_options.ParseFrameVariants:
            self.__parse_frame_variants(parsed_text=parsed_text,
                                        frame_variant_collection=self._parse_options.FrameVariantsCollection)

        return parsed_text

    # region private methods

    # TODO. In further, this is considered to be departed from base text parser
    # TODO. and treated as an (optional) element of the text processing pipeline.
    @staticmethod
    def __to_lemmas(locale_mods, parsed_text):
        assert(issubclass(locale_mods, BaseLanguageMods))
        return [locale_mods.replace_specific_word_chars(lemma) if isinstance(lemma, str) else lemma
                for lemma in parsed_text.iter_terms(term_format=TermFormat.Lemma)]

    # TODO. In further, this is considered to be departed from base text parser
    # TODO. and treated as an (optional) element of the text processing pipeline.
    @staticmethod
    def __parse_frame_variants(parsed_text, frame_variant_collection):
        """ Parsing frame variants in doc sentences.
        """
        assert(isinstance(parsed_text, BaseParsedText))
        assert(isinstance(frame_variant_collection, FrameVariantsCollection) or
               frame_variant_collection is None)

        if frame_variant_collection is None:
            return

        objs_it = FrameVariantsParser.iter_frames_from_lemmas(
            frame_variants=frame_variant_collection,
            lemmas=BaseTextParser.__to_lemmas(locale_mods=RussianLanguageMods,
                                              parsed_text=parsed_text))

        parsed_text.modify_by_bounded_objects(
            modified_objs=objs_it,
            get_obj_bound_func=lambda variant: variant.get_bound())

    # endregion
