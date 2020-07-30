# -*- coding: utf-8 -*-
import collections
import logging

from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.frame_variants.parse import FrameVariantsParser
from arekit.common.news.base import News
from arekit.common.news.parse_options import NewsParseOptions
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.utils import split_by_whitespaces
from arekit.processing.text.news_stat import debug_statistics, debug_show_terms
from arekit.processing.text.parsed import ParsedText
from arekit.processing.text.tokens import Tokens
from arekit.processing.text.token import Token


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextParser:
    """
    Represents a parser of news sentences.
    As a result we have a list of TERMS, where term could be a
        1) Word
        2) Token
    """

    def __init__(self):
        pass

    @staticmethod
    def parse_news(news, parse_options):
        assert(isinstance(news, News))
        assert(isinstance(parse_options, NewsParseOptions))

        parsed_sentences = []

        return_text = True if not parse_options.ParseEntities else False
        for sentence in news.iter_sentences(return_text=return_text):
            if parse_options.ParseEntities:
                text_with_entities = news.EntitiesParser.parse(sentence)
                parsed_sentence = TextParser.__parse_string_list(string_iter=text_with_entities,
                                                                 stemmer=parse_options.Stemmer)
            else:
                parsed_sentence = TextParser.parse(text=sentence,
                                                   stemmer=parse_options.Stemmer)

            parsed_sentences.append(parsed_sentence)

        parsed_news = ParsedNews(news_id=news.ID,
                                 parsed_sentences=parsed_sentences)

        NewsTermsShow = False
        NewsTermsStatisticShow = False

        if NewsTermsStatisticShow:
            debug_statistics(parsed_news)
        if NewsTermsShow:
            debug_show_terms(parsed_news)

        if parse_options.ParseFrameVariants:
            TextParser.__post_processing(parsed_news=parsed_news,
                                         frame_variant_collection=parse_options.FrameVariantsCollection)

        return ParsedNews(news_id=news.ID,
                          parsed_sentences=parsed_sentences)

    @staticmethod
    def parse(text, stemmer=None):
        assert(isinstance(text, unicode))
        terms = TextParser.__parse_core(text)
        return ParsedText(terms, stemmer=stemmer)

    @staticmethod
    def __post_processing(parsed_news, frame_variant_collection):
        """
        Labeling frame variants in doc sentences.
        """
        assert(isinstance(parsed_news, ParsedNews))
        assert(isinstance(frame_variant_collection, FrameVariantsCollection) or frame_variant_collection is None)

        if frame_variant_collection is None:
            return

        parsed_news.modify_parsed_sentences(
            lambda sentence: FrameVariantsParser.parse_frames_in_parsed_text(
                frame_variants_collection=frame_variant_collection,
                parsed_text=sentence))

    # region private methods

    @staticmethod
    def __parse_string_list(string_iter, stemmer=None):
        assert(isinstance(string_iter, collections.Iterable))

        terms = []
        for text in string_iter:
            if not isinstance(text, unicode):
                terms.append(text)
                continue
            new_terms = TextParser.__parse_core(text)
            terms.extend(new_terms)

        return ParsedText(terms, stemmer=stemmer)

    @staticmethod
    def __parse_core(text, keep_tokens=True):
        """
        Separates sentence into list of parsed_news

        save_tokens: bool
            keep token information in result list of parsed_news.
        return: list
            list of unicode parsed_news, where each term: word or token
        """
        assert(isinstance(text, unicode))
        assert(isinstance(keep_tokens, bool))

        terms = TextParser.__process_words(words=split_by_whitespaces(text),
                                           keep_tokens=keep_tokens)

        TextParser.__log_debug(terms)

        return terms

    @staticmethod
    def __process_words(words, keep_tokens):
        """
        parsed_news: list
            list of parsed_news
        keep_tokes: bool
            keep or remove tokens from list of parsed_news
        """
        assert(isinstance(words, list))
        parsed = []
        for word in words:

            if word is None:
                continue

            words_and_tokens = TextParser.__split_tokens(word)

            if not keep_tokens:
                words_and_tokens = [word for word in words_and_tokens if not isinstance(word, Token)]

            parsed.extend(words_and_tokens)

        return parsed

    @staticmethod
    def __split_tokens(term):
        """
        Splitting off tokens from parsed_news ending, i.e. for example:
            term: "сказать,-" -> "(term: "сказать", ["COMMA_TOKEN", "DASH_TOKEN"])
        return: (unicode or None, list)
            modified term and list of extracted tokens.
        """

        url = Tokens.try_create_url(term)
        if url is not None:
            return [url]

        l = 0
        words_and_tokens = []
        while l < len(term):

            # Token.
            token = Tokens.try_create(term[l])
            if token is not None:
                if token.get_token_value() != Tokens.NEW_LINE:
                    words_and_tokens.append(token)
                l += 1

            # Number.
            elif unicode.isdigit(term[l]):
                k = l + 1
                while k < len(term) and unicode.isdigit(term[k]):
                    k += 1
                token = Tokens.try_create_number(term[l:k])
                assert(token is not None)
                words_and_tokens.append(token)
                l = k

            # Term.
            else:
                k = l + 1
                while k < len(term):
                    token = Tokens.try_create(term[k])
                    if token is not None and token.get_token_value() != Tokens.DASH:
                        break
                    k += 1
                words_and_tokens.append(term[l:k])
                l = k

        return words_and_tokens

    @staticmethod
    def __try_term_as_token(term):
        url = Tokens.try_create_url(term)
        if url is not None:
            return url
        number = Tokens.try_create_number(term)
        if number is not None:
            return number
        return Tokens.try_create(term)

    @staticmethod
    def __log_debug(terms):
        for term in terms:
            if isinstance(term, Token):
                logger.debug(u'"TOKEN: {}, {}" '.format(
                    term.get_meta_value(),
                    term.get_token_value()))
            else:
                logger.debug(u'"WORD: {}" '.format(term))

    # endregion
