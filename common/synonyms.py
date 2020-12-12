import collections

from arekit.common import log_utils


class SynonymsCollection(object):

    def __init__(self, iter_group_values_lists, is_read_only, debug):
        assert(isinstance(iter_group_values_lists, collections.Iterable))
        assert(isinstance(is_read_only, bool))
        assert(isinstance(debug, bool))

        # Assumes to be filled
        self.__by_synonym = {}
        self.__by_index = []

        self.__is_read_only = is_read_only
        self.__debug = debug
        self.__fill(iter_grop_values_lists=iter_group_values_lists)

    # region properties

    @property
    def IsReadOnly(self):
        return self.__is_read_only

    # endregion

    # region public 'add' methods

    def add_synonym_value(self, value):
        assert(isinstance(value, unicode))

        if self.contains_synonym_value(value):
            raise Exception((u"Collection already contains synonyms '{}'".format(value)).encode('utf-8'))

        if self.__is_read_only:
            raise Exception((u"Failed to add '{}'. Synonym collection is read only!".format(value)).encode('utf-8'))

        synonym_id = self.create_synonym_id(value)
        self.__by_synonym[synonym_id] = self.__get_groups_count()
        self.__by_index.append([value])

    # endregion

    # region public 'contains' methods

    def contains_synonym_value(self, value):
        return self.__contains_synonym_value(value)

    # endregion

    # region public 'get' methods

    def get_synonym_group_index(self, value):
        assert(isinstance(value, unicode))
        return self.__get_group_index(value)

    def try_get_synonym_group_index(self, value, default=-1):
        return self.__get_group_index(value) if self.__contains_synonym_value(value) else default

    # endregion

    # region public 'create' methods

    def create_synonym_id(self, value):
        raise NotImplementedError()

    # endregion

    # region public 'iter' methods

    def iter_synonym_values(self, value):
        assert(isinstance(value, unicode))
        id = self.create_synonym_id(value)
        index = self.__by_synonym[id]
        return iter(self.__by_index[index])

    def iter_by_index(self):
        return iter(self.__by_index)

    def iter_group(self, group_index):
        assert(isinstance(group_index, int))
        return iter(self.__by_index[group_index])

    # endregion

    # region private methods

    def __fill(self, iter_grop_values_lists):
        for group in iter_grop_values_lists:
            self.__process_group(group)

    def __process_group(self, group_values_list):
        group_index = len(self.__by_index)
        synonym_list = []

        for synonym_value in group_values_list:

            value = synonym_value.strip()

            synonym_id = self.create_synonym_id(value)

            if synonym_id in self.__by_synonym and self.__debug:
                log_utils.log_synonym_existed(value)
                continue

            synonym_list.append(value)
            self.__by_synonym[synonym_id] = group_index

        self.__by_index.append(synonym_list)

    def __get_groups_count(self):
        return len(self.__by_index)

    def __get_group_index(self, value):
        synonym_id = self.create_synonym_id(value)
        return self.__by_synonym[synonym_id]

    def __contains_synonym_value(self, value):
        return self.create_synonym_id(value) in self.__by_synonym

    # endregion

    # region overridden methods

    def __len__(self):
        return len(self.__by_index)

    # endregion
