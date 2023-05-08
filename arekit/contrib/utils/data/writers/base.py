class BaseWriter(object):

    def open_target(self, target):
        pass

    def commit_line(self, storage):
        pass

    def close_target(self):
        pass

    def write_all(self, storage, target):
        """ Performs the writing process of the whole storage.
            The implementation and support of the related operation
            may vary and depends on the nature of the storage, which
            briefly might keep all the data in memory (available)
            or cache only temporary information (unavailable)

            storage: BaseRowsStorage
            target: str
        """
        raise NotImplementedError()
