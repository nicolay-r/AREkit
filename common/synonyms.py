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

    @property
    def IsReadOnly(self):
        return self.__is_read_only

    @property
    def Stemmer(self):
        return self.__stemmer

    def add_synonym(self, s):
        assert(isinstance(s, unicode))
        assert(not self.has_synonym(s))
        assert(not self.__is_read_only)
        id = self.create_synonym_id(self.__stemmer, s)
        self.__by_synonym[id] = self._get_groups_count()
        self.__by_index.append([s])

    def has_synonym(self, s):
        assert(isinstance(s, unicode))
        id = self.create_synonym_id(self.__stemmer, s)
        return id in self.__by_synonym

    def get_synonyms_list(self, s):
        assert(isinstance(s, unicode))
        id = self.create_synonym_id(self.__stemmer, s)
        index = self.__by_synonym[id]
        return self.__by_index[index]

    def get_synonym_group_index(self, s):
        assert(isinstance(s, unicode))
        return self._get_group_index(s)

    def _get_groups_count(self):
        return len(self.__by_index)

    def _get_group_index(self, s):
        id = self.create_synonym_id(self.__stemmer, s)
        return self.__by_synonym[id]

    #TODO: Deprecated. Use iter instead
    def get_group_by_index(self, index):
        assert(isinstance(index, int))
        return self.__by_index[index]

    @staticmethod
    def create_synonym_id(stemmer, s):
        return stemmer.lemmatize_to_str(s)

    def iter_by_index(self):
        for item in self.__by_index:
            yield item

    def iter_group(self, group_index):
        assert(isinstance(group_index, int))
        for item in self.__by_index[group_index]:
            yield item

    def __len__(self):
        return len(self.__by_index)