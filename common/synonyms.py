from arekit.processing.lemmatization.base import Stemmer


# TODO. Make stemmer as an optional parameter.
# TODO. Ommit the lemmatization if the stemmer is None.
class SynonymsCollection(object):

    def __init__(self, by_index, by_synonym, stemmer, is_read_only):
        assert(isinstance(by_index, list))
        assert(isinstance(by_synonym, dict))
        assert(isinstance(stemmer, Stemmer))
        self.__by_index = by_index
        self.__by_synonym = by_synonym
        # TODO. stemmer should be a part of the nested RuSentRel Synonyms Collection.
        self.__stemmer = stemmer
        self.__is_read_only = is_read_only

    # region properties

    @property
    def IsReadOnly(self):
        return self.__is_read_only

    # TODO. This should be a part of the nested RuSentRel Synonyms Collection.
    @property
    def Stemmer(self):
        return self.__stemmer

    # endregion

    # region public 'add' methods

    def add_synonym_value(self, value):
        assert(isinstance(value, unicode))

        if self.contains_synonym_value(value):
            raise Exception((u"Collection already contains synonyms '{}'".format(value)).encode('utf-8'))

        if self.__is_read_only:
            raise Exception((u"Failed to add '{}'. Synonym collection is read only!".format(value)).encode('utf-8'))

        id = self.create_synonym_id(self.__stemmer, value)
        self.__by_synonym[id] = self.__get_groups_count()
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

    # TODO. This should be a part of the nested RuSentRel Synonyms Collection.
    @staticmethod
    def create_synonym_id(stemmer, value):
        assert(isinstance(stemmer, Stemmer))
        return stemmer.lemmatize_to_str(value)

    # endregion

    # region public 'iter' methods

    def iter_synonym_values(self, value):
        assert(isinstance(value, unicode))
        id = self.create_synonym_id(self.__stemmer, value)
        index = self.__by_synonym[id]
        for v in self.__by_index[index]:
            yield v

    def iter_by_index(self):
        for item in self.__by_index:
            yield item

    def iter_group(self, group_index):
        assert(isinstance(group_index, int))
        for item in self.__by_index[group_index]:
            yield item

    # endregion

    # region private methods

    def __get_groups_count(self):
        return len(self.__by_index)

    def __get_group_index(self, value):
        id = self.create_synonym_id(self.__stemmer, value)
        return self.__by_synonym[id]

    def __contains_synonym_value(self, value):
        id = self.create_synonym_id(self.__stemmer, value)
        return id in self.__by_synonym

    # endregion

    # region overridden methods

    def __len__(self):
        return len(self.__by_index)

    # endregion
