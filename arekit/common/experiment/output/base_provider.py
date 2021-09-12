class BaseOutputProvider(object):
    """
    Provider, internally based on pandas dataframe.
    """

    @property
    def DataFrame(self):
        raise NotImplementedError()

    def load(self, source):
        raise NotImplementedError()
