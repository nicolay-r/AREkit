class BaseCVFolding(object):
    """
    Default CV splitter
    """

    def __init__(self):
        """
        cv_count: int
            is an amount of folds to be produced by an algorithm
        cv_index: int
            is an iteration index of a Cross-Fold validation.
        """
        self.__cv_count = None
        self.__iteration_index = 0

    # region Properties

    @property
    def CVCount(self):
        return self.__cv_count

    @property
    def IterationIndex(self):
        return self.__iteration_index

    # endregion

    def set_cv_count(self, value):
        assert(isinstance(value, int))
        self.__cv_count = value

    def set_iteration_index(self, value):
        assert(isinstance(value, int))
        assert(value < self.__cv_count)
        self.__iteration_index = value

    def get_cv_train_test_pair_by_index(self, doc_ids_iter):
        raise NotImplementedError()
