from arekit.processing.lemmatization.base import Stemmer


class RuSentRelNewsParseOptions(object):

    def __init__(self, stemmer, keep_tokens):
        assert(isinstance(stemmer, Stemmer) or isinstance(stemmer, type(None)))
        assert(isinstance(keep_tokens, bool))
        self.__stemmer = stemmer
        self.__keep_tokens = keep_tokens

    @property
    def Stemmer(self):
        return self.__stemmer

    @property
    def KeepTokens(self):
        return self.__keep_tokens
