import random

from arekit.contrib.utils.cv.splitters.base import CrossValidationSplitter


class SimpleCrossValidationSplitter(CrossValidationSplitter):
    """ This splitter assumes to performs folding
        without extra additional statistics of the related documents.
    """

    def __init__(self, shuffle=True, seed=1):
        self.__shuffle = shuffle
        self.__seed = seed

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

    # endregion

    def items_to_cv_pairs(self, doc_ids, cv_count):
        """
        Splits array of indices into list of pairs (train_indices_list,
        test_indices_list)
        """
        assert(isinstance(doc_ids, set))
        assert(isinstance(cv_count, int))

        doc_ids_list = list(doc_ids)

        if self.__shuffle:
            random.Random(self.__seed).shuffle(doc_ids_list)

        chunks = self.__chunk_it(doc_ids_list, cv_count)

        for test_index, chunk in enumerate(chunks):
            train_indices = list(range(len(chunks)))
            train_indices.remove(test_index)

            large = [v for train_index in train_indices for v in chunks[train_index]]
            small = chunk

            yield large, small
