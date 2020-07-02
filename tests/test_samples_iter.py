#!/usr/bin/python
import csv
import gzip


def iter_tsv_gzip(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with gzip.open(input_file, mode="rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            yield line


for line in iter_tsv_gzip(input_file=u"test_data/sample_train.tsv.gz"):
    for a in line:
        print a,
    print


