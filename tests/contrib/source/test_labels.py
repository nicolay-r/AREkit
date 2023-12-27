import sys
import unittest

sys.path.append('../')

from arekit.common.labels.base import Label, NoLabel
from labels import NegativeLabel, PositiveLabel


class TestLabels(unittest.TestCase):

    def test(self):

        pos_label = PositiveLabel()
        neg_label = NegativeLabel()
        no_label = NoLabel()
        label = Label()

        self.assertTrue(no_label == no_label)
        self.assertFalse(pos_label == neg_label)
        self.assertFalse(pos_label == label)


if __name__ == '__main__':
    unittest.main()
