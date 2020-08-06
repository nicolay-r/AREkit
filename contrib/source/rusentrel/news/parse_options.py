from arekit.common.news.parse_options import NewsParseOptions
from arekit.processing.lemmatization.base import Stemmer


class RuSentRelNewsParseOptions(NewsParseOptions):

    def __init__(self, stemmer, frame_variants_collection):
        assert(isinstance(stemmer, Stemmer) or isinstance(stemmer, type(None)))

        super(RuSentRelNewsParseOptions, self).__init__(
            parse_entities=True,
            stemmer=stemmer,
            frame_variants_collection=frame_variants_collection)
