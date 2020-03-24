import random

from arekit.contrib.experiments.cv.base import BaseCVFolding


class SimpleCVFolding(BaseCVFolding):

    def __init__(self,  doc_ids):
        assert(isinstance(doc_ids, list))
        super(SimpleCVFolding, self).__init__()

        self.__doc_ids = doc_ids

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

    def __items_to_cv_pairs(self, shuffle=True, seed=1):
        """
        Splits array of indices into list of pairs (train_indices_list,
        test_indices_list)
        """

        if shuffle:
            random.Random(seed).shuffle(self.__doc_ids)

        chunks = self.__chunk_it(self.__doc_ids, self.__cv_count)

        for test_index, chunk in enumerate(chunks):
            train_indices = range(len(chunks))
            train_indices.remove(test_index)

            train = [v for train_index in train_indices for v in chunks[train_index]]
            test = chunk

            yield train, test

    # endregion

    def get_cv_pair_by_index(self):
        it = self.__items_to_cv_pairs(shuffle=True, seed=1)

        for index, pair in enumerate(it):
            train, test = pair
            if index == self.IterationIndex:
                return train, test
