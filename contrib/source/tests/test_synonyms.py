# -*- coding: utf-8 -*-
import unittest

from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.tests.utils import read_rusentrel_synonyms_collection


class TestSynonymsCollection(unittest.TestCase):

    __version = RuSentRelVersions.V11

    def test(self):
        synonyms = read_rusentrel_synonyms_collection(self.__version)

        for group_index in xrange(len(synonyms)):
            for group_item in synonyms.iter_group(group_index):
                print group_item
            print "---"

    def test_iter_by_index(self):
        synonyms = read_rusentrel_synonyms_collection(self.__version)

        for item in synonyms.iter_by_index():
            assert(isinstance(item, list))
            for word in item:
                print word
            print "==="

    def test_iter_synonym_values(self):
        searching_value = u'америка'
        synonyms = read_rusentrel_synonyms_collection(self.__version)
        print u"Request: {}".format(searching_value)
        print u"-------"
        for value in synonyms.iter_synonym_values(value=searching_value):
            print value


if __name__ == '__main__':
    unittest.main()
