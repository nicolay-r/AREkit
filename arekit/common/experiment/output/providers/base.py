class BaseOutputProvider(object):
    """
    Provider, internally based on pandas dataframe.
    """

    # TODO. 206. Remove (this is a part of the storage)
    @property
    def DataFrame(self):
        raise NotImplementedError()

    def load(self, source):
        raise NotImplementedError()
