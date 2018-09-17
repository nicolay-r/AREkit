# -*- coding: utf-8 -*-
from core.source.tokens import Tokens


class TextParser:
    """
    Represents a parser of news sentences.
    Now uses in neural networks for text processing.
    As a result we have a list of terms, where term could be a
        1) Word
        2) Token
    """

    def __init__(self):
        pass

    @staticmethod
    def parse_to_list(text, save_tokens=False, debug=False):
        """
        Separates sentence into list of terms
        return: list
            list of unicode terms, where each term: word or token
        """
        assert(isinstance(text, unicode))

        result_terms = []

        for term in [w.strip() for w in text.split(' ')]:

            if term is None:
                continue

            modified_term, tokens = TextParser._split_tokens(term)

            if modified_term is not None:
                result_terms.append(modified_term)

            if len(tokens) > 0 and save_tokens:
                result_terms.extend(tokens)

        if debug:
            TextParser._print(result_terms)

        return result_terms

    @staticmethod
    def _split_tokens(term):
        """
        Splitting off tokens from terms ending, i.e. for example:
            term: "сказать,-" -> "(term: "сказать", ["COMMA_TOKEN", "DASH_TOKEN"])
        return: (unicode or None, list)
            modified term and list of extracted tokens.
        """

        tokens_before = []
        tokens_after = []

        number = Tokens.try_create_number(term)
        if number is not None:
            return None, [number]

        first_index = 0
        last_index = len(term) - 1

        while first_index <= last_index:
            token = Tokens.try_create(term[first_index])
            if token is None:
                break
            tokens_before.append(token)
            first_index += 1

        while last_index >= first_index:
            token = Tokens.try_create(term[last_index])
            if token is None:
                break
            tokens_after.append(token)
            last_index -= 1

        # inplace concat.
        tokens_before.extend(tokens_after[::-1])

        modified_term = term[first_index:last_index+1] if last_index >= first_index else None
        return modified_term, tokens_before

    @staticmethod
    def _print(terms):
        for t in terms:
            print '"{}" '.format(t.encode('utf-8')),
