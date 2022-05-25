from arekit.common.news.parser import NewsParser
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.rusentrel.news_reader import RuSentRelNewsReader
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


def init_rusentrel_doc(doc_id, text_parser, synonyms):
    assert(isinstance(doc_id, int))
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(synonyms, SynonymsCollection))

    news = RuSentRelNewsReader.read_document(doc_id=doc_id,
                                             synonyms=synonyms)

    parsed_news = NewsParser.parse(news=news,
                                   text_parser=text_parser)

    opinions = RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id)

    collection = OpinionCollection(opinions=opinions,
                                   synonyms=synonyms,
                                   error_on_duplicates=True,
                                   error_on_synonym_end_missed=True)

    return news, parsed_news, collection
