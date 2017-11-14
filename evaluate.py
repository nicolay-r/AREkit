#!/usr/bin/python
# -*- coding: utf-8 -*-

import io_utils
from core.eval import Evaluator


def show(r):
    print "pos_prec: %.4f, neg_prec: %.4f" % (r['pos_prec'], r['neg_prec'])
    print "pos_recall: %.4f, neg_recall: %.4f" % (r['pos_recall'], r['neg_recall'])
    print "f1_pos: %.4f, f1_neg: %.4f" % (r['f1_pos'], r['f1_neg'])
    print "f1: %.4f" % ((r['f1_pos'] + r['f1_neg']) / 2)

e = Evaluator(
        io_utils.get_synonyms_filepath(),
        io_utils.test_root(),
        io_utils.get_etalon_root())

r = e.evaluate(io_utils.test_indices())
show(r)
