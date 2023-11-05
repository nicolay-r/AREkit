from collections.abc import Iterable


class LinkedDataWrapper(object):

    def __init__(self, linked_data):
        assert(isinstance(linked_data, Iterable))
        self.__linked_data = list(linked_data)
        self.__tag = None

    @property
    def RelatedDocID(self):
        """ Linked data is limited to the particular document.
        """
        raise NotImplementedError()

    @property
    def First(self):
        return self[0]

    @property
    def Tag(self):
        return self.__tag

    def set_tag(self, value):
        self.__tag = value

    def _get_data_label(self, item):
        raise NotImplementedError()

    def iter_labels(self):
        for item in self.__linked_data:
            yield self._get_data_label(item)

    def __getitem__(self, item):
        assert(isinstance(item, int))
        return self.__linked_data[item]

    def __iter__(self):
        for data in self.__linked_data:
            yield data

    def __len__(self):
        return len(self.__linked_data)
