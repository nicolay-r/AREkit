import collections


class LinkedDataWrapper(object):

    def __init__(self, linked_data):
        assert(isinstance(linked_data, collections.Iterable))
        self.__linked_data = list(linked_data)

    @property
    def _LikedData(self):
        return self.__linked_data

    def __getitem__(self, item):
        assert(isinstance(item, int))
        return self.__linked_data[item]

    def __iter__(self):
        for text_opinion in self.__linked_data:
            yield text_opinion

    def __len__(self):
        return len(self.__linked_data)
