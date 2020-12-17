import sys
import unittest

sys.path.append('../')

from arekit.common.labels.base import PositiveLabel, NegativeLabel, Label, NeutralLabel


class TestLabels(unittest.TestCase):

    def test(self):

        pos_label = PositiveLabel()
        neg_label = NegativeLabel()
        neutral_label = NeutralLabel()
        label = Label()

        print neutral_label == neutral_label


if __name__ == '__main__':
    unittest.main()
