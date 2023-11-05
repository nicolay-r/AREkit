from collections.abc import Iterable

from arekit.common import log_utils


class SynonymsCollection(object):

    def __init__(self, iter_group_values_lists=None, is_read_only=True, debug=False):
        """ iter_group_values_lists: iterable or None
            is_read_only: bool
                whether the relation collection could be expanded or not
            debug: bool
                utilized for logging the salient information during usage.
        """
        assert(isinstance(iter_group_values_lists, Iterable) or iter_group_values_lists is None)
        assert(isinstance(is_read_only, bool))
        assert(isinstance(debug, bool))

        # Assumes to be filled
        self.__by_sid = {}
        self.__by_index = []

        self.__is_read_only = is_read_only
        self.__debug = debug
        self.__fill(iter_group_values_lists=[] if iter_group_values_lists is None else iter_group_values_lists)

    # region properties

    @property
    def IsReadOnly(self):
        return self.__is_read_only

    # endregion

    # region public 'add' methods

    def add_synonym_value(self, value):
        assert(isinstance(value, str))

        if self.__contains_synonym_value(value):
            raise Exception(("Collection already contains synonyms '{}'".format(value)).encode('utf-8'))

        if self.__is_read_only:
            raise Exception(("Failed to add '{}'. Synonym collection is read only!".format(value)).encode('utf-8'))

        sid = self._create_external_sid(value)
        self.__by_sid[sid] = self.__get_groups_count()
        self.__by_index.append([value])

    # endregion

    # region public 'contains' methods

    def contains_synonym_value(self, value):
        return self.__contains_synonym_value(value)

    # endregion

    # region public 'get' methods

    def get_synonym_group_index(self, value):
        """ NOTE: Before use this, please take a look at the grouping (see #327 issue).
            It is better to use that class API rather than pass that method for `value_to_group_id_func`
        """
        assert(isinstance(value, str))
        return self.__get_group_index(value)

    # endregion

    # region public 'create' methods

    def create_synonym_id(self, value):
        return self._create_external_sid(value)

    # endregion

    # region protected methods

    def _contains_sid(self, v_id):
        return v_id in self.__by_sid

    def _create_internal_sid(self, value):
        """ Utilized during filling stage.
        """
        raise NotImplementedError()

    def _create_external_sid(self, value):
        raise NotImplementedError()

    # endregion

    # region public 'iter' methods

    def iter_synonym_values(self, value):
        assert(isinstance(value, str))
        sid = self._create_external_sid(value)
        index = self.__by_sid[sid]
        return iter(self.__by_index[index])

    def iter_by_index(self):
        return iter(self.__by_index)

    def iter_group(self, group_index):
        assert(isinstance(group_index, int))
        return iter(self.__by_index[group_index])

    # endregion

    # region private methods

    def __fill(self, iter_group_values_lists):
        for group in iter_group_values_lists:
            self.__process_group(group)

    def __process_group(self, group_values_list):
        group_index = len(self.__by_index)
        synonym_list = []

        for synonym_value in group_values_list:

            value = synonym_value.strip()

            sid = self._create_internal_sid(value)

            if self._contains_sid(sid) and self.__debug:
                log_utils.log_synonym_existed(value)
                continue

            synonym_list.append(value)
            self.__by_sid[sid] = group_index

        self.__by_index.append(synonym_list)

    def __get_groups_count(self):
        return len(self.__by_index)

    def __get_group_index(self, value):
        sid = self._create_external_sid(value)
        return self.__by_sid[sid]

    def __contains_synonym_value(self, value):
        return self._contains_sid(self._create_external_sid(value))

    # endregion

    # region overridden methods

    def __len__(self):
        return len(self.__by_index)

    # endregion