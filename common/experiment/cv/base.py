class BaseCVFolding(object):
    """ Default, abstract CV splitter
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

    def _items_to_cv_pairs(self, doc_ids):
        """ Provides pairs for every cv-iteration.
        """
        raise NotImplementedError()

    def set_cv_count(self, value):
        assert(isinstance(value, int))
        self.__cv_count = value

    def set_iteration_index(self, value):
        assert(isinstance(value, int))
        assert(value < self.__cv_count)
        self.__iteration_index = value

    def get_cv_split(self, doc_ids_iter, data_types):
        assert(isinstance(data_types, list))

        if len(data_types) > 2:
            raise NotImplementedError(u"Experiments with such amount of data-types are not supported!")

        if len(data_types) == 1:
            # By default we provide the same output since
            # there is no need to perform splitting onto single part
            return {
                data_types[0]: doc_ids_iter
            }

        it = self._items_to_cv_pairs(set(doc_ids_iter))

        for index, pair in enumerate(it):
            large, small = pair
            if index == self.IterationIndex:
                return {
                    data_types[0]: large,
                    data_types[1]: small
                }
