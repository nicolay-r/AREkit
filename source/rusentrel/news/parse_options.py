from arekit.common.news_parse_options import NewsParseOptions
from arekit.processing.lemmatization.base import Stemmer


class RuSentRelNewsParseOptions(NewsParseOptions):

    def __init__(self, stemmer, keep_tokens, frame_variants_collection):
        assert(isinstance(stemmer, Stemmer) or isinstance(stemmer, type(None)))
        assert(isinstance(keep_tokens, bool))

        super(RuSentRelNewsParseOptions, self).__init__(
            parse_entities=True,
            stemmer=stemmer,
            frame_variants_collection=frame_variants_collection)

        self.__keep_tokens = keep_tokens

    @property
    # TODO. Tokens hiding actually discarded
    def KeepTokens(self):
        return self.__keep_tokens
