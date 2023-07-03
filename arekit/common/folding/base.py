class BaseDataFolding(object):
    """ Describes and provides API on how to handle doc_ids during experiment,
        i.e. how many states does nested folding algorithm supports,
        how to perform folding for a particular state (current),
        and how to such state into string.
    """

    def __init__(self, supported_data_types=None):
        assert(isinstance(supported_data_types, list) or supported_data_types is None)
        self._supported_data_types = supported_data_types

    def fold_doc_ids_set(self, doc_ids):
        """ Perform the doc_ids folding process onto provided data_types
        """
        raise NotImplementedError()
