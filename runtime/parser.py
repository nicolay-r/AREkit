from core.source.tokens import Tokens, Token
from core.processing.lemmatization.base import Stemmer


# TODO. Move into processing/text directory.
class TextParser:
    """
    Represents a parser of news sentences.
    Now uses in neural networks for text processing.
    As a result we have a list of TERMS, where term could be a
        1) Word
        2) Token
    """

    def __init__(self):
        pass

    @staticmethod
    def parse(text, stemmer=None, debug=False):
        terms = TextParser.__parse_core(text, debug=debug)
        return ParsedText(terms, stemmer=stemmer)

    @staticmethod
    def from_string(str, separator=' ', stemmer=None):

        def __term_or_token(term):
            token = TextParser.__try_term_as_token(term)
            return token if token is not None else term

        assert(isinstance(str, str))
        terms = [word.strip(' ') for word in str.split(separator)]
        terms = [__term_or_token(t) for t in terms]
        return ParsedText(terms, stemmer=stemmer)

    @staticmethod
    def __parse_core(text, debug=False):
        """
        Separates sentence into list of terms

        return: list
            list of unicode terms, where each term: word or token
        """
        assert(isinstance(text, str))

        words = [word.strip(' ') for word in text.split(' ')]
        terms = TextParser.__process_words(words)

        if debug:
            TextParser.__print(terms)

        return terms

    @staticmethod
    def __process_words(words):
        """
        terms: list
            list of terms
        keep_tokes: bool
            keep or remove tokens from list of terms
        """
        assert(isinstance(words, list))
        parsed = []
        for word in words:

            if word is None:
                continue

            words_and_tokens = TextParser.__split_tokens(word)

            parsed.extend(words_and_tokens)

        return parsed

    @staticmethod
    def __split_tokens(term):
        """
        Splitting off tokens from terms ending, i.e. for example:
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
            # token
            token = Tokens.try_create(term[l])
            if token is not None:
                if token.get_token_value() != Tokens.NEW_LINE:
                    words_and_tokens.append(token)
                l += 1
            # number
            elif str.isdigit(term[l]):
                k = l + 1
                while k < len(term) and str.isdigit(term[k]):
                    k += 1
                token = Tokens.try_create_number(term[l:k])
                assert(token is not None)
                words_and_tokens.append(token)
                l = k
            # term
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
    def __print(terms):
        for term in terms:
            if isinstance(term, Token):
                print(('"TOKEN: {}, {}" '.format(
                    term.get_original_value(),
                    term.get_token_value())))
            else:
                print(('"WORD: {}" '.format(term)))

# TODO. Move into processing/text directory.
class ParsedText:
    """
    Represents a processed text with extra parameters
    that were used during parsing.
    """

    def __init__(self, terms, stemmer=None):
        assert(isinstance(terms, list))
        assert(isinstance(stemmer, Stemmer) or stemmer is None)
        self.__terms = terms
        self.__lemmas = None
        if stemmer is not None:
            self.__lemmatize(stemmer)

    def iter_original_terms(self):
        for term in self.__terms:
            yield self.__output_term(term)

    def is_term(self, index):
        assert(isinstance(index, int))
        return isinstance(self.__terms[index], str)

    def iter_lemmas(self):
        for lemma in self.__lemmas:
            yield self.__output_term(lemma)

    def iter_raw_terms(self):
        for term in self.__terms:
            yield term

    def iter_raw_lemmas(self):
        for lemma in self.__lemmas:
            yield lemma

    def __lemmatize(self, stemmer):
        """
        Compose a list of lemmatized versions of terms
        PS: Might be significantly slow, depending on stemmer were used.
        """
        assert(isinstance(stemmer, Stemmer))
        self.__lemmas = ["".join(stemmer.lemmatize_to_list(t)) if isinstance(t, str) else t
                         for t in self.__terms]

    def to_string(self):
        return ' '.join(self.iter_original_terms())

    @staticmethod
    def __output_term(term, tokens_as_meta=False):
        return ParsedText.__get_token_as_term(term, tokens_as_meta) if isinstance(term, Token) else term

    @staticmethod
    def __get_token_as_term(token, hide):
        return token.get_token_value() if hide else token.get_original_value()

    def __len__(self):
        return len(self.__terms)

    def __iter__(self):
        for term in self.__terms:
            yield term
