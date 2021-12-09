import collections

from arekit.common.entities.base import Entity
from arekit.common.frame_variants.collection import FrameVariantsCollection

from arekit.common.languages.mods import BaseLanguageMods
from arekit.common.languages.ru.mods import RussianLanguageMods
from arekit.common.news.base import News
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.text.options import NewsParseOptions
from arekit.common.text.parsed import BaseParsedText
from arekit.processing.frames.parser import FrameVariantsParser
from arekit.processing.text.enums import TermFormat


class BaseTextParser(object):

    def __init__(self, create_parsed_text_func=None):
        """
        create_parsed_text_func: is a function with the following signature:
            (terms, options) -> ParsedText
        """
        assert(callable(create_parsed_text_func) or create_parsed_text_func is None)

        if create_parsed_text_func is None:
            # default implementation
            create_parsed_text_func = lambda terms, _: BaseParsedText(terms=terms)

        self.__create_parsed_text_func = create_parsed_text_func

    def parse_news(self, news, parse_options):
        assert(isinstance(news, News))
        assert(isinstance(parse_options, NewsParseOptions))

        parsed_sentences = [self.__parse_sentence(news, sent_ind, parse_options)
                            for sent_ind in range(news.SentencesCount)]

        parsed_news = ParsedNews(doc_id=news.ID,
                                 parsed_sentences=parsed_sentences)

        if parse_options.ParseFrameVariants:
            self.__parse_frame_variants(parsed_news=parsed_news,
                                        frame_variant_collection=parse_options.FrameVariantsCollection)

        return parsed_news

    # region protected abstract

    def _parse_to_tokens_list(self, text, keep_tokens=True):
        raise NotImplementedError()

    # endregion

    # region private methods

    @staticmethod
    def __to_lemmas(locale_mods, parsed_text):
        assert(issubclass(locale_mods, BaseLanguageMods))
        return [locale_mods.replace_specific_word_chars(lemma) if isinstance(lemma, str) else lemma
                for lemma in parsed_text.iter_terms(term_format=TermFormat.Lemma)]

    @staticmethod
    def __parse_frame_variants(parsed_news, frame_variant_collection):
        """
        Labeling frame variants in doc sentences.
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(frame_variant_collection, FrameVariantsCollection) or
               frame_variant_collection is None)

        if frame_variant_collection is None:
            return

        parsed_news.modify_parsed_sentences(
            sentence_objs_upd_func=lambda sentence:
            FrameVariantsParser.iter_frames_from_lemmas(
                frame_variants=frame_variant_collection,
                lemmas=BaseTextParser.__to_lemmas(locale_mods=RussianLanguageMods,
                                                  parsed_text=sentence)),
            get_obj_bound_func=lambda variant: variant.get_bound())

    def __parse_sentence(self, news, sent_ind, parse_options):
        assert(isinstance(news, News))
        assert(isinstance(parse_options, NewsParseOptions))

        if parse_options.ParseEntities:
            # Providing a modified list with parsed unicode terms.
            terms_list = news.sentence_to_terms_list(sent_ind)
            return self.__parse_terms_list(terms_iter=terms_list,
                                           skip_term=lambda term: isinstance(term, Entity),
                                           parse_options=parse_options)

        # Processing the ordinary sentence text.
        sentence = news.iter_sentences()
        return self.__parse(text=sentence.Text,
                            parse_options=parse_options)

    def __parse_terms_list(self, terms_iter, skip_term, parse_options):
        assert(isinstance(terms_iter, collections.Iterable))
        assert(isinstance(parse_options, NewsParseOptions))
        assert(callable(skip_term))

        processed_terms = []
        for term in terms_iter:

            if skip_term(term):
                processed_terms.append(term)
                continue

            new_terms = self._parse_to_tokens_list(term)
            processed_terms.extend(new_terms)

        return self.__create_parsed_text_func(processed_terms, parse_options)

    def __parse(self, text, parse_options):
        assert(isinstance(text, str))
        assert(isinstance(parse_options, NewsParseOptions))
        return self.__create_parsed_text_func(self._parse_to_tokens_list(text=text), parse_options)

    # endregion
