import collections

from arekit.common.entities.base import Entity
from arekit.common.frames.variants.collection import FrameVariantsCollection

from arekit.common.languages.mods import BaseLanguageMods
from arekit.common.languages.ru.mods import RussianLanguageMods
from arekit.common.text.options import TextParseOptions
from arekit.common.text.parsed import BaseParsedText
from arekit.processing.frames.parser import FrameVariantsParser
from arekit.processing.text.enums import TermFormat


class BaseTextParser(object):

    def __init__(self, parse_options, create_parsed_text_func=None):
        """
        create_parsed_text_func: is a function with the following signature:
            (terms, options) -> ParsedText
        """
        assert(callable(create_parsed_text_func) or create_parsed_text_func is None)
        assert(isinstance(parse_options, TextParseOptions))

        if create_parsed_text_func is None:
            # default implementation
            create_parsed_text_func = lambda terms, _: BaseParsedText(terms=terms)

        self.__create_parsed_text_func = create_parsed_text_func
        self._parse_options = parse_options

    def parse(self, terms_list):
        """
        terms_list: list of terms
        returns:
            ParsedText
        """
        assert(isinstance(terms_list, list))

        # Tokenization stage. (PPL 1).
        parsed_text = self.__tokenize_terms(terms_list)

        # Frames parsing stage. (PPL 2).
        if self._parse_options.ParseFrameVariants:
            self.__parse_frame_variants(parsed_text=parsed_text,
                                        frame_variant_collection=self._parse_options.FrameVariantsCollection)

        return parsed_text

    # region protected abstract

    def _parse_to_tokens_list(self, text):
        raise NotImplementedError()

    # endregion

    # region private methods

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

    # TODO. In further, this is considered to be departed from base text parser
    # TODO. and treated as an (optional) element of the text processing pipeline.
    def __tokenize_terms(self, terms_list):
        assert(isinstance(terms_list, list))

        handled_terms = self.__handle_terms(
            terms_iter=terms_list,
            skip_term=lambda term: isinstance(term, Entity),
            term_handler=lambda term: self._parse_to_tokens_list(term))

        # Do parsing.
        return self.__create_parsed_text_func(handled_terms,
                                              self._parse_options)

    @staticmethod
    def __handle_terms(terms_iter, skip_term, term_handler):
        assert(isinstance(terms_iter, collections.Iterable))
        assert(callable(skip_term))

        processed_terms = []
        for term in terms_iter:

            if skip_term(term):
                processed_terms.append(term)
                continue

            new_terms = term_handler(term)
            processed_terms.extend(new_terms)

        return processed_terms

    # endregion
