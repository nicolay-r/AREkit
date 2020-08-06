from arekit.common.synonyms import SynonymsCollection
from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.text.parser import TextParser
from arekit.source.rusentrel.news.base import RuSentRelNews
from arekit.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection


def init_rusentrel_doc(doc_id, stemmer, synonyms, unique_frame_variants):
    assert(isinstance(doc_id, int))
    assert(isinstance(stemmer, Stemmer))
    assert(isinstance(synonyms, SynonymsCollection))

    news = RuSentRelNews.read_document(doc_id=doc_id,
                                       synonyms=synonyms)

    options = RuSentRelNewsParseOptions(stemmer=stemmer,
                                        keep_tokens=True,
                                        frame_variants_collection=unique_frame_variants)

    parsed_news = TextParser.parse_news(news, options)

    opinions = RuSentRelOpinionCollection.load_collection(doc_id=doc_id,
                                                          synonyms=synonyms)
    return news, parsed_news, opinions