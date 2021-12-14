import logging

from arekit.common.text.parser import BaseTextParser
from arekit.common.utils import split_by_whitespaces
from arekit.processing.text.parsed import DefaultParsedText
from arekit.processing.text.tokens import Tokens
from arekit.processing.text.token import Token


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DefaultTextParser(BaseTextParser):
    """
    Default parser implementation.
    """

    def __init__(self, parse_options):
        super(DefaultTextParser, self).__init__(
            create_parsed_text_func=lambda terms, options: DefaultParsedText(terms=terms, stemmer=options.Stemmer),
            parse_options=parse_options)

    # region protected methods

    def _parse_to_tokens_list(self, text):
        """
        Separates sentence into list

        save_tokens: bool
            keep token information in result list of parsed_news.
        return: list
            list of unicode parsed_news, where each term: word or token
        """
        assert(isinstance(text, str))

        terms = self._process_words(words=split_by_whitespaces(text))

        return terms

    # endregion

    # region private static methods

    def _process_words(self, words):
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

            processed = self._process_words_to_terms_list(word=word)

            parsed.extend(processed)

        return parsed

    def _process_words_to_terms_list(self, word):

        words_and_tokens = DefaultTextParser.__split_tokens(word)

        if not self._parse_options.KeepTokens:
            words_and_tokens = [word for word in words_and_tokens if not isinstance(word, Token)]

        return words_and_tokens

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
            elif str.isdigit(term[l]):
                k = l + 1
                while k < len(term) and str.isdigit(term[k]):
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
