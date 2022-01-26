import collections


class BaseDataFolding(object):
    """ Describes and provides API on how to handle doc_ids during experiment,
        i.e. how many states does nested folding algorithm supports,
        how to perform folding for a particular state (current),
        and how to such state into string.
    """

    def __init__(self, doc_ids_to_fold, supported_data_types, states_count):
        assert(isinstance(doc_ids_to_fold, collections.Iterable))
        assert(isinstance(supported_data_types, list))
        assert(isinstance(states_count, int) and states_count > 0)
        self._doc_ids_to_fold_set = set(doc_ids_to_fold)
        self._supported_data_types = supported_data_types
        self._states = states_count
        self._state_index = 0

    @property
    def StatesCount(self):
        return self._states

    @property
    def StateIndex(self):
        return self._state_index

    @property
    def Name(self):
        raise NotImplementedError()

    def _assign_index(self, i):
        self._state_index = i

    def contains_doc_id(self, doc_id):
        assert(isinstance(doc_id, int))
        return doc_id in self._doc_ids_to_fold_set

    def iter_doc_ids(self):
        return iter(self._doc_ids_to_fold_set)

    def iter_supported_data_types(self):
        """ Iterates through data_types, supported in a related experiment
            Note:
            In CV-split algorithm, the first part corresponds to a LARGE split,
            Jand second to small; therefore, the correct sequence is as follows:
            DataType.Train, DataType.Test.
        """
        return iter(self._supported_data_types)

    def iter_states(self):
        """ Performs iteration over states supported by folding algorithm
            Default:
                considering a single state.
        """
        for state_index in range(self._states):
            self._assign_index(state_index)
            yield None

    def fold_doc_ids_set(self):
        """ Perform the doc_ids folding process onto provided data_types
        """
        raise NotImplementedError()
