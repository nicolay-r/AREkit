#!/usr/bin/python
import csv
import gzip
import sys

sys.path.append('../../')

from arekit.contrib.networks.sample import InputSample


def iter_tsv_gzip(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with gzip.open(input_file, mode="rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            yield line


# TODO. This should be initialized.
# TODO. Processing before.
words_vocab = None
config = None

for line in iter_tsv_gzip(input_file=u"test_data/sample_train.tsv.gz"):
    s = InputSample.from_tsv_row(text_opinion_id=line[0],
                                 terms=line[1].split(u' '),
                                 subj_ind=line[2],
                                 obj_ind=line[3],
                                 words_vocab=words_vocab,
                                 config=None)


