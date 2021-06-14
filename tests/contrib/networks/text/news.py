from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.source.rusentrel.news.base import RuSentRelNews
from arekit.contrib.source.rusentrel.news.parse_options import RuSentRelNewsParseOptions
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.processing.lemmatization.base import Stemmer
from arekit.processing.text.parser import TextParser


def init_rusentrel_doc(doc_id, stemmer, synonyms, unique_frame_variants):
    assert(isinstance(doc_id, int))
    assert(isinstance(stemmer, Stemmer))
    assert(isinstance(synonyms, SynonymsCollection))

    news = RuSentRelNews.read_document(doc_id=doc_id,
                                       synonyms=synonyms)

    options = RuSentRelNewsParseOptions(stemmer=stemmer,
                                        frame_variants_collection=unique_frame_variants)

    parsed_news = TextParser.parse_news(news, options)

    opinions = RuSentRelOpinionCollection.iter_opinions_from_doc(doc_id=doc_id)

    collection = OpinionCollection(opinions=opinions,
                                   synonyms=synonyms,
                                   error_on_duplicates=True,
                                   error_on_synonym_end_missed=True)

    return news, parsed_news, collection
