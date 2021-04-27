import sys
import unittest

sys.path.append('../')

from arekit.common.labels.base import Label, NeutralLabel
from arekit.contrib.source.common.labels import NegativeLabel, PositiveLabel


class TestLabels(unittest.TestCase):

    def test(self):

        pos_label = PositiveLabel()
        neg_label = NegativeLabel()
        neutral_label = NeutralLabel()
        label = Label()

        self.assertTrue(neutral_label == neutral_label)
        self.assertFalse(pos_label == neg_label)
        self.assertFalse(pos_label == label)


if __name__ == '__main__':
    unittest.main()
