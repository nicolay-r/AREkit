from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


def init_rusentrel_doc(doc_id, text_parser, parse_options, synonyms):
    assert(isinstance(doc_id, int))
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(synonyms, SynonymsCollection))

    news = RuSentRelNews.read_document(doc_id=doc_id,
                                       synonyms=synonyms)

    parsed_news = text_parser.parse_news(news, parse_options)

    opinions = RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id)

    collection = OpinionCollection(opinions=opinions,
                                   synonyms=synonyms,
                                   error_on_duplicates=True,
                                   error_on_synonym_end_missed=True)

    return news, parsed_news, collection
