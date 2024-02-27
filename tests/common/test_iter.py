import unittest

from arekit.common.pipeline.utils import BatchIterator


class TestBalancing(unittest.TestCase):

    def test(self):
        batch_it = BatchIterator(data_iter=iter(range(10)), batch_size=3)
        for a in batch_it:
            print(a)
