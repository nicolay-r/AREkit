import unittest

from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions


class TestRuSentRel(unittest.TestCase):

    rsr_version = RuSentRelVersions.V11

    def test_iter_train_indices(self):
        train_indices = list(RuSentRelIOUtils.iter_train_indices(self.rsr_version))
        for i in train_indices:
            print i,

        for i in range(1, 46):

            if i in [9, 22, 26]:
                continue

            self.assertIn(i, train_indices)

    def test_iter_test_indices(self):
        test_indices = list(RuSentRelIOUtils.iter_test_indices(self.rsr_version))

        for i in range(46, 76):
            if i in [70]:
                continue

            self.assertIn(i, test_indices)


if __name__ == '__main__':
    unittest.main()
