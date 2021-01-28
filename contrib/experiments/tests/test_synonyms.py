# -*- coding: utf-8 -*-
import unittest

from arekit.contrib.experiments.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.lemmatization.mystem import MystemWrapper


class TestSynonymsCollection(unittest.TestCase):

    __rusentrel_version = RuSentRelVersions.V11
    __ruattittudes_version = RuAttitudesVersions.V20Large

    def __read_synonyms_collection(self):
        # Initializing stemmer
        stemmer = MystemWrapper()

        # Reading synonyms collection.
        return RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer,
                                                                   version=self.__rusentrel_version)

    def test(self):
        synonyms = self.__read_synonyms_collection()

        for group_index in xrange(len(synonyms)):
            for group_item in synonyms.iter_group(group_index):
                print group_item
            print "---"

    def test_iter_by_index(self):
        synonyms = self.__read_synonyms_collection()

        for item in synonyms.iter_by_index():
            assert(isinstance(item, list))
            for word in item:
                print word
            print "==="

    def test_iter_synonym_values(self):
        searching_value = u'америка'
        synonyms = self.__read_synonyms_collection()
        print u"Request: {}".format(searching_value)
        print u"-------"
        for value in synonyms.iter_synonym_values(value=searching_value):
            print value


if __name__ == '__main__':
    unittest.main()
