from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.common.text.stemmer import Stemmer

from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


def init_rusentrel_doc(doc_id, text_parser, stemmer, synonyms, unique_frame_variants):
    assert(isinstance(doc_id, int))
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(stemmer, Stemmer))
    assert(isinstance(synonyms, SynonymsCollection))

    news = RuSentRelNews.read_document(doc_id=doc_id,
                                       synonyms=synonyms,
                                       version=RuSentRelVersions.V11)

    options = RuSentRelNewsParseOptions(stemmer=stemmer,
                                        frame_variants_collection=unique_frame_variants)

    parsed_news = text_parser.parse_news(news, options)

    opins_it = RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id)
    opinions = OpinionCollection(opinions=opins_it,
                                 synonyms=synonyms,
                                 error_on_synonym_end_missed=True,
                                 error_on_duplicates=True)

    return news, parsed_news, opinions
