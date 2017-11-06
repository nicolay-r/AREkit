import io
import core.environment as env


class SynonymsCollection:

    def __init__(self, by_index):
        self.by_index = by_index
        self.by_synonym = self._index_by_synonyms()
        print "==========================================="
        for key, value in self.by_synonym.iteritems():
            print "'{}', {}".format(key.encode('utf-8'), value)
        print "==========================================="

    @staticmethod
    def from_file(filepath):
        by_index = []
        with io.open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                synonyms = []

                args = line.split(',')
                for s in args:
                    value = env.stemmer.lemmatize_to_str(s.strip())
                    synonyms.append(value)

                by_index.append(synonyms)
        return SynonymsCollection(by_index)

    def has_synonym(self, synonym):
        assert(type(synonym) == unicode)
        synonym = env.stemmer.lemmatize_to_str(synonym)
        return synonym in self.by_synonym

    def get_synonyms(self, synonym):
        assert(type(synonym) == unicode)
        synonym = env.stemmer.lemmatize_to_str(synonym)
        index = self.by_synonym[synonym]
        return self.by_index[index]

    def _index_by_synonyms(self):
        index = {}
        for i, synonyms in enumerate(self.by_index):
            for s in synonyms:
                index[s] = i
        return index
