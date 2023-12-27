from arekit.common.docs.parser import DocumentParser
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.base import SynonymsCollection
from arekit.common.text.parser import BaseTextParser

from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter
from arekit.contrib.source.rusentrel.docs_reader import RuSentRelDocumentsReader
from arekit.contrib.source.rusentrel.opinions.collection import RuSentRelOpinions
from labels import PositiveLabel, NegativeLabel


def init_rusentrel_doc(doc_id, text_parser, synonyms):
    assert(isinstance(doc_id, int))
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(synonyms, SynonymsCollection))

    doc = RuSentRelDocumentsReader.read_document(doc_id=doc_id,
                                                 synonyms=synonyms,
                                                 version=RuSentRelVersions.V11)

    parsed_doc = DocumentParser.parse(doc=doc, text_parser=text_parser)

    opins_it = RuSentRelOpinions.iter_from_doc(
        doc_id=doc_id,
        labels_fmt=RuSentRelLabelsFormatter(pos_label_type=PositiveLabel,
                                            neg_label_type=NegativeLabel))
    opinions = OpinionCollection(opinions=opins_it,
                                 synonyms=synonyms,
                                 error_on_synonym_end_missed=True,
                                 error_on_duplicates=True)

    return doc, parsed_doc, opinions
