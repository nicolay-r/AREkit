import collections
import random

from arekit.common.experiment.cv.base import BaseCVFolding


class SimpleCVFolding(BaseCVFolding):
    """ This folding algorithm assumes to performs folding
        without extra additional statistics of the related documents.
    """

    def __init__(self):
        super(SimpleCVFolding, self).__init__()

    # region private methods

    @staticmethod
    def __chunk_it(sequence, num):
        avg = len(sequence) / float(num)
        out = []
        last = 0.0

        while last < len(sequence):
            out.append(sequence[int(last):int(last + avg)])
            last += avg

        return out

    def __items_to_cv_pairs(self, doc_ids, shuffle=True, seed=1):
        """
        Splits array of indices into list of pairs (train_indices_list,
        test_indices_list)
        """
        assert(isinstance(doc_ids, list))

        if shuffle:
            random.Random(seed).shuffle(doc_ids)

        chunks = self.__chunk_it(doc_ids, self.CVCount)

        for test_index, chunk in enumerate(chunks):
            train_indices = range(len(chunks))
            train_indices.remove(test_index)

            train = [v for train_index in train_indices for v in chunks[train_index]]
            test = chunk

            yield train, test

    # endregion

    def get_cv_train_test_pair_by_index(self, doc_ids_iter):
        assert(isinstance(doc_ids_iter, collections.Iterable))

        it = self.__items_to_cv_pairs(
            doc_ids=list(doc_ids_iter),
            shuffle=True,
            seed=1)

        for index, pair in enumerate(it):
            train, test = pair
            if index == self.IterationIndex:
                return train, test
