import csv
import gzip
import sys
import unittest

sys.path.append('../../../')

from arekit.common.utils import split_by_whitespaces
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
        config.modify_terms_per_context(35)

        samples = []
        for line in self.iter_tsv_gzip(input_file=samples_filepath):
            _id, label, text, subj_ind, obj_ind = line

            _id = _id.decode('utf-8')
            text = text.decode('utf-8')

            print u"------------------"
            print u"INPUT SAMPLE DATA"
            print u"------------------"
            print u"id: {}".format(_id)
            print u"label: {}".format(label)
            print u"terms: {}".format(text)
            print u"subj_ind: {}".format(subj_ind)
            print u"obj_ind: {}".format(obj_ind)

            s = InputSample.from_tsv_row(input_sample_id=_id,
                                         terms=split_by_whitespaces(text),
                                         subj_ind=int(subj_ind),
                                         obj_ind=int(obj_ind),
                                         words_vocab=words_vocab,
                                         config=config)

            print u"------------------"
            print u"NETWORK INPUT DATA"
            print u"------------------"
            for key, value in s:
                print key, value

            samples.append(s)


if __name__ == '__main__':
    unittest.main()