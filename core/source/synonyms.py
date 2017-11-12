import io
import core.env as env


class SynonymsCollection:

    def __init__(self, by_index, by_synonym):
        assert(type(by_index) == list)
        assert(type(by_synonym) == dict)
        self.by_index = by_index
        self.by_synonym = by_synonym

    @staticmethod
    def from_file(filepath):
        by_index = []
        by_synonym = {}

        with io.open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for group_index, line in enumerate(lines):
                synonym_list = []
                args = line.split(',')

                for s in args:
                    value = s.strip()
                    id = SynonymsCollection._create_synonym_id(value)

                    if id in by_synonym:
                        print "Collection already has a value '{}'. Skipped".format(value)
                        continue

                    synonym_list.append(value)
                    by_synonym[id] = group_index

                by_index.append(synonym_list)

        return SynonymsCollection(by_index, by_synonym)

    def add_synonym(self, s):
        assert(type(s) == unicode)
        assert(not self.has_synonym(s))
        id = self._create_synonym_id(s)
        self.by_synonym[id] = self._get_groups_count()
        self.by_index.append([s])

    def has_synonym(self, s):
        assert(type(s) == unicode)
        id = self._create_synonym_id(s)
        return id in self.by_synonym

    def get_synonyms_list(self, s):
        assert(type(s) == unicode)
        id = self._create_synonym_id(s)
        index = self.by_synonym[id]
        return self.by_index[index]

    def get_synonym_group_index(self, s):
        assert(type(s) == unicode)
        return self._get_group_index(s)

    def _get_groups_count(self):
        return len(self.by_index)

    def _get_group_index(self, s):
        id = self._create_synonym_id(s)
        return self.by_synonym[id]

    @staticmethod
    def _create_synonym_id(s):
        return env.stemmer.lemmatize_to_str(s)
