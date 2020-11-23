# -*- coding: utf-8 -*-
import collections
import logging

from arekit.common.entities.base import Entity
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.frame_variants.parse import FrameVariantsParser
from arekit.common.news.base import News
from arekit.common.news.parse_options import NewsParseOptions
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.utils import split_by_whitespaces
from arekit.processing.text.parsed import ParsedText
from arekit.processing.text.tokens import Tokens
from arekit.processing.text.token import Token


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextParser:

    def __init__(self):
        pass

    @staticmethod
    def parse_news(news, parse_options):
        assert(isinstance(news, News))
        assert(isinstance(parse_options, NewsParseOptions))

        parsed_sentences = [TextParser.__parse_sentence(news, sent_ind, parse_options)
                            for sent_ind in range(news.SentencesCount)]

        parsed_news = ParsedNews(news_id=news.ID,
                                 parsed_sentences=parsed_sentences)

        if parse_options.ParseFrameVariants:
            TextParser.__parse_frame_variants(parsed_news=parsed_news,
                                              frame_variant_collection=parse_options.FrameVariantsCollection)

        return parsed_news

    # region private methods

    @staticmethod
    def __parse(text, stemmer=None):
        assert(isinstance(text, unicode))
        terms = TextParser.__parse_core(text)
        return ParsedText(terms, stemmer=stemmer)

    @staticmethod
    def __parse_sentence(news, sent_ind, parse_options):
        assert(isinstance(news, News))
        assert(isinstance(parse_options, NewsParseOptions))

        if parse_options.ParseEntities:
            # Providing a modified list with parsed unicode terms.
            terms_list = news.parse_sentence(sent_ind)
            return TextParser.__parse_string_list(terms_iter=terms_list,
                                                  skip_term=lambda term: isinstance(term, Entity),
                                                  stemmer=parse_options.Stemmer)

        # Processing the ordinary sentence text.
        sentence = news.iter_sentences(sent_ind)
        return TextParser.__parse(text=sentence.Text,
                                  stemmer=parse_options.Stemmer)

    @staticmethod
    def __parse_frame_variants(parsed_news, frame_variant_collection):
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

    @staticmethod
    def __parse_string_list(terms_iter, skip_term, stemmer=None):
        assert(isinstance(terms_iter, collections.Iterable))
        assert(callable(skip_term))

        processed_terms = []
        for term in terms_iter:

            if skip_term(term):
                processed_terms.append(term)
                continue

            new_terms = TextParser.__parse_core(term)
            processed_terms.extend(new_terms)

        return ParsedText(terms=processed_terms, stemmer=stemmer)

    @staticmethod
    def __parse_core(text, keep_tokens=True):
        """
        Separates sentence into list

        save_tokens: bool
            keep token information in result list of parsed_news.
        return: list
            list of unicode parsed_news, where each term: word or token
        """
        assert(isinstance(text, unicode))
        assert(isinstance(keep_tokens, bool))

        terms = TextParser.__process_words(words=split_by_whitespaces(text),
                                           keep_tokens=keep_tokens)

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

    # endregion
