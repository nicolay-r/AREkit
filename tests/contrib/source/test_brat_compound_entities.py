import unittest

from arekit.common.bound import Bound
from arekit.common.entities.collection import EntityCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.source.brat.entities.compound import BratCompoundEntity
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.brat.sentences_reader import BratDocumentSentencesReader
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


class TestCompoundEntites(unittest.TestCase):

    text = "мама мыла раму"
    entities = [
        BratEntity(id_in_doc="T1", e_type="PERSON", index_begin=0, index_end=4, value="мама"),
        BratEntity(id_in_doc="T2", e_type="VERB", index_begin=5, index_end=8, value="мыл"),
        BratEntity(id_in_doc="T3", e_type="OBJECT", index_begin=9, index_end=13, value="раму"),
        BratEntity(id_in_doc="T3", e_type="ACTION", index_begin=0, index_end=8, value="мама мыл")
    ]

    def test(self):
        s_data = [
            {"text": self.text, "ind_begin": 0, "ind_end": len(self.text)}
        ]

        synonyms = StemmerBasedSynonymCollection(stemmer=MystemWrapper(), is_read_only=False, debug=False)

        collection = EntityCollection(
            self.entities,
            value_to_group_id_func=lambda value:
                SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(synonyms, value))

        sentences = BratDocumentSentencesReader.from_sentences_data(entities=collection,
                                                                    sentences_data=s_data)

        for sentence in sentences:
            for e, b in sentence.iter_entity_with_local_bounds():
                assert(isinstance(b, Bound))
                print(type(e))
                print("{} ({}, {})".format(e.Value, b.Position, b.Position + b.Length))
                if isinstance(e, BratCompoundEntity):
                    for ee in e.iter_childs():
                        print("\t{} ({}, {}) [{}]".format(ee.Value, ee.IndexBegin, ee.IndexEnd, ee.Type))


if __name__ == '__main__':
    unittest.main()
