from core.processing.lemmatization.base import Stemmer


class SynonymsCollection(object):

    def __init__(self, by_index, by_synonym, stemmer, is_read_only):
        assert(isinstance(by_index, list))
        assert(isinstance(by_synonym, dict))
        assert(isinstance(stemmer, Stemmer))
        self.__by_index = by_index
        self.__by_synonym = by_synonym
        self.__stemmer = stemmer
        self.__is_read_only = is_read_only

    # region properties

    @property
    def IsReadOnly(self):
        return self.__is_read_only

    @property
    def Stemmer(self):
        return self.__stemmer

    # endregion

    # region public 'add' methods

    def add_synonym(self, s):
        assert(isinstance(s, unicode))
        assert(not self.contains_synonym(s))
        assert(not self.__is_read_only)
        id = self.create_synonym_id(self.__stemmer, s)
        self.__by_synonym[id] = self.__get_groups_count()
        self.__by_index.append([s])

    # endregion

    # region public 'contains' methods

    def contains_synonym(self, s):
        assert(isinstance(s, unicode))
        id = self.create_synonym_id(self.__stemmer, s)
        return id in self.__by_synonym

    # endregion

    # region public 'get' methods

    def get_synonyms_list(self, s):
        assert(isinstance(s, unicode))
        id = self.create_synonym_id(self.__stemmer, s)
        index = self.__by_synonym[id]
        return self.__by_index[index]

    def get_synonym_group_index(self, s):
        assert(isinstance(s, unicode))
        return self.__get_group_index(s)

    # endregion

    # region public 'create' methods

    @staticmethod
    def create_synonym_id(stemmer, s):
        return stemmer.lemmatize_to_str(s)

    # endregion

    # region public 'iter' methods

    def iter_by_index(self):
        for item in self.__by_index:
            yield item

    def iter_group(self, group_index):
        assert(isinstance(group_index, int))
        for item in self.__by_index[group_index]:
            yield item

    # endregion

    # region overriden methods

    def __len__(self):
        return len(self.__by_index)

    # endregion

    # region private methods

    def __get_groups_count(self):
        return len(self.__by_index)

    def __get_group_index(self, s):
        id = self.create_synonym_id(self.__stemmer, s)
        return self.__by_synonym[id]

    # endregion

