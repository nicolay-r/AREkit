class BaseExperimentDataFolding(object):
    """ Describes and provides API on how to handle doc_ids during experiment,
        i.e. how many states does nested folding algorithm supports,
        how to perform folding for a particular state (current),
        and how to such state into string.
    """

    def __init__(self, doc_ids_to_fold, supported_data_types):
        assert(isinstance(doc_ids_to_fold, list))
        assert(isinstance(supported_data_types, list))
        self._doc_ids_to_fold_set = set(doc_ids_to_fold)
        self._supported_data_types = supported_data_types
        self._iteration_index = 0

    @property
    def Name(self):
        return

    @property
    def IterationIndex(self):
        return self._iteration_index

    def contains_doc_id(self, doc_id):
        assert(isinstance(doc_id, int))
        return doc_id in self._doc_ids_to_fold_set

    def iter_doc_ids(self):
        return iter(self._doc_ids_to_fold_set)

    def iter_states(self):
        """ Performs iteration over states supported by folding algorithm
            Default:
                considering a single state.
        """
        yield None

    def iter_supported_data_types(self):
        """ Iterates through data_types, supported in a related experiment
            Note:
            In CV-split algorithm, the first part corresponds to a LARGE split,
            Jand second to small; therefore, the correct sequence is as follows:
            DataType.Train, DataType.Test.
        """
        return iter(self._supported_data_types)

    def fold_doc_ids_set(self):
        """ Perform the doc_ids folding process onto provided data_types
        """
        raise NotImplementedError()

    def get_current_state(self):
        """ Provides the name of the current state.
        """
        raise NotImplementedError()
