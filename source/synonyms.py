import io
from core.processing.lemmatization.base import Stemmer


class SynonymsCollection:

    def __init__(self, by_index, by_synonym, stemmer):
        assert(isinstance(by_index, list))
        assert(isinstance(by_synonym, dict))
        assert(isinstance(stemmer, Stemmer))
        self.by_index = by_index
        self.by_synonym = by_synonym
        self.stemmer = stemmer

    @classmethod
    def from_file(cls, filepath, stemmer, debug=False):
        assert(isinstance(filepath, unicode))
        assert(isinstance(stemmer, Stemmer))
        by_index = []
        by_synonym = {}
        SynonymsCollection._from_file(filepath, by_index, by_synonym, stemmer, debug)
        return cls(by_index, by_synonym, stemmer)


    @classmethod
    def from_files(cls, filepaths, stemmer, debug=False):
        assert(isinstance(filepaths, list))
        assert(isinstance(stemmer, Stemmer))
        by_index = []
        by_synonym = {}
        for filepath in filepaths:
            SynonymsCollection._from_file(filepath, by_index, by_synonym, stemmer, debug)
        return cls(by_index, by_synonym, stemmer)


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
                        print "Collection already has a value '{}'. Skipped".format(value.encode('utf-8'))
                        continue

                    synonym_list.append(value)
                    by_synonym[id] = group_index

                by_index.append(synonym_list)

    def add_synonym(self, s):
        assert(isinstance(s, unicode))
        assert(not self.has_synonym(s))
        id = self._create_synonym_id(self.stemmer, s)
        self.by_synonym[id] = self._get_groups_count()
        self.by_index.append([s])

    def has_synonym(self, s):
        assert(isinstance(s, unicode))
        id = self._create_synonym_id(self.stemmer, s)
        return id in self.by_synonym

    def get_synonyms_list(self, s):
        assert(isinstance(s, unicode))
        id = self._create_synonym_id(self.stemmer, s)
        index = self.by_synonym[id]
        return self.by_index[index]

    def get_synonym_group_index(self, s):
        assert(isinstance(s, unicode))
        return self._get_group_index(s)

    def _get_groups_count(self):
        return len(self.by_index)

    def _get_group_index(self, s):
        id = self._create_synonym_id(self.stemmer, s)
        return self.by_synonym[id]

    def get_group_by_index(self, index):
        assert(isinstance(index, int))
        return self.by_index[index]

    @staticmethod
    def _create_synonym_id(stemmer, s):
        return stemmer.lemmatize_to_str(s)
