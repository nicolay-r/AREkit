import io
from core.processing.lemmatization.base import Stemmer


class SynonymsCollection:

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

    @classmethod
    def from_file(cls, filepath, stemmer, is_read_only=True, debug=False):
        assert(isinstance(filepath, str))
        assert(isinstance(stemmer, Stemmer))
        by_index = []
        by_synonym = {}
        SynonymsCollection._from_file(filepath, by_index, by_synonym, stemmer, debug)
        return cls(by_index=by_index,
                   by_synonym=by_synonym,
                   stemmer=stemmer,
                   is_read_only=is_read_only)

    @staticmethod
    def _from_file(filepath, by_index, by_synonym, stemmer, debug):
        """
        reading from 'filepath' and initialize 'by_index' and 'by_synonym'
        structures
        by_index: list
            to be initialized
        by_synonym: dict
            to be initialized
        stemmer: Stemmer
            stemmer
        returns: None
        """
        assert(isinstance(by_index, list))
        assert(isinstance(by_synonym, dict))
        assert(isinstance(stemmer, Stemmer))

        with io.open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                group_index = len(by_index)
                synonym_list = []
                args = line.split(',')

                for s in args:
                    value = s.strip()
                    id = SynonymsCollection._create_synonym_id(stemmer, value)

                    if id in by_synonym and debug:
                        print(("Collection already has a value '{}'. Skipped".format(value)))
                        continue

                    synonym_list.append(value)
                    # print 'adding:', id, '->', group_index
                    by_synonym[id] = group_index

                by_index.append(synonym_list)

    def add_synonym(self, s, lemmatize=True):
        assert(isinstance(s, str))
        assert(not self.has_synonym(s))
        assert(not self.__is_read_only)
        id = self._create_synonym_id(self.__stemmer, s, lemmatize=lemmatize)
        self.__by_synonym[id] = self._get_groups_count()
        self.__by_index.append([s])

    def has_synonym(self, s, lemmatize=True):
        assert(isinstance(s, str))
        id = self._create_synonym_id(self.__stemmer, s, lemmatize=lemmatize)
        return id in self.__by_synonym

    def get_synonyms_list(self, s):
        assert(isinstance(s, str))
        id = self._create_synonym_id(self.__stemmer, s)
        index = self.__by_synonym[id]
        return self.__by_index[index]

    def get_synonym_group_index(self, s, lemmatize=True):
        assert(isinstance(s, str))
        return self._get_group_index(s, lemmatize=lemmatize)

    def _get_groups_count(self):
        return len(self.__by_index)

    def _get_group_index(self, s, lemmatize):
        id = self._create_synonym_id(self.__stemmer, s, lemmatize=lemmatize)
        return self.__by_synonym[id]

    def get_group_by_index(self, index):
        assert(isinstance(index, int))
        return self.__by_index[index]

    @staticmethod
    def _create_synonym_id(stemmer, s, lemmatize=True):
        if lemmatize:
            return stemmer.lemmatize_to_str(s)
        else:
            return s

    def iter_by_index(self):
        for item in self.__by_index:
            yield item

    def iter_group(self, group_index):
        assert(isinstance(group_index, int))
        for item in self.__by_index[group_index]:
            yield item

    def __len__(self):
        return len(self.__by_index)
