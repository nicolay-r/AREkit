import collections


class LinkedDataWrapper(object):

    def __init__(self, linked_data):
        assert(isinstance(linked_data, collections.Iterable))
        self.__linked_data = list(linked_data)

    @property
    def First(self):
        return self[0]

    @property
    def _LikedData(self):
        return self.__linked_data

    @staticmethod
    def _aggregate_by_first(item, label):
        raise NotImplementedError()

    def aggregate_data(self, label):
        return self._aggregate_by_first(item=self.First, label=label)

    def __getitem__(self, item):
        assert(isinstance(item, int))
        return self.__linked_data[item]

    def __iter__(self):
        for data in self.__linked_data:
            yield data

    def __len__(self):
        return len(self.__linked_data)
