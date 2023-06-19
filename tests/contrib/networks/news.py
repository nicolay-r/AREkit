from arekit.common.docs.parser import DocumentParser
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.base import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.news_reader import RuSentRelDocumentsReader
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinions
from tests.contrib.networks.labels import TestPositiveLabel, TestNegativeLabel


def init_rusentrel_doc(doc_id, text_parser, synonyms):
    assert(isinstance(doc_id, int))
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(synonyms, SynonymsCollection))

    news = RuSentRelDocumentsReader.read_document(doc_id=doc_id,
                                                  synonyms=synonyms)

    parsed_news = DocumentParser.parse(news=news,
                                       text_parser=text_parser)

    opinions = RuSentRelOpinions.iter_from_doc(
        doc_id=doc_id,
        labels_fmt=RuSentRelLabelsFormatter(pos_label_type=TestPositiveLabel,
                                            neg_label_type=TestNegativeLabel)
    )

    collection = OpinionCollection(opinions=opinions,
                                   synonyms=synonyms,
                                   error_on_duplicates=True,
                                   error_on_synonym_end_missed=True)

    return news, parsed_news, collection
