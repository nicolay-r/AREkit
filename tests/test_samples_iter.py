#!/usr/bin/python
import csv
import gzip
import sys
import unittest

sys.path.append('../../')

from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.sample import InputSample


class TestSamplesIteration(unittest.TestCase):

    @staticmethod
    def iter_tsv_gzip(input_file):
        """Reads a tab separated value file."""
        with gzip.open(input_file, mode="rt") as f:
            reader = csv.reader(f, delimiter="\t")
            for line in reader:
                yield line

    @staticmethod
    def read_vocab(input_file):
        words = {}
        with gzip.open(input_file, mode="rt") as f:
            for w_ind, line in enumerate(f.readlines()):
                w = line.decode('utf-8').strip()
                if w in words:
                    raise Exception(u"Word already presented: {}".format(w).encode('utf-8'))
                words[w] = w_ind
        return words

    def test(self):

        vocab_filepath = u"test_data/vocab.txt.gz"
        samples_filepath = u"test_data/sample_train.tsv.gz"

        words_vocab = self.read_vocab(vocab_filepath)
        config = DefaultNetworkConfig()

        samples = []
        for line in self.iter_tsv_gzip(input_file=samples_filepath):
            _id, label, terms, subj_ind, obj_ind = line

            terms = terms.decode('utf-8')

            print u"id: {}".format(_id)
            print u"label: {}".format(label)
            print u"terms: {}".format(terms)
            print u"subj_ind: {}".format(subj_ind)
            print u"obj_ind: {}".format(obj_ind)

            s = InputSample.from_tsv_row(row_id=_id,
                                         terms=terms.split(u' '),
                                         subj_ind=int(subj_ind),
                                         obj_ind=int(obj_ind),
                                         words_vocab=words_vocab,
                                         config=config)

            samples.append(s)


if __name__ == '__main__':
    unittest.main()
