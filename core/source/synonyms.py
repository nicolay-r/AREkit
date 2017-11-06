import io
import core.environment as env


class SynonymsCollection:

    def __init__(self, by_index, by_synonym):
        self.by_index = by_index
        self.by_synonym = by_synonym
        # print "*******************************************"
        # for key, value in self.by_synonym.iteritems():
        #     print "'{}', {}".format(key.encode('utf-8'), value)
        # print "*******************************************"

    @staticmethod
    def from_file(filepath):
        by_index = []
        by_synonym = {}
        with io.open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for group_index, line in enumerate(lines):
                synonyms = []
                args = line.split(',')
                for s in args:
                    value = s.strip()
                    synonyms.append(value)
                    if value not in by_synonym:
                        by_synonym[env.stemmer.lemmatize_to_str(value)] = group_index
                    else:
                        print "Collection already has a value '{}'".format(value)

                by_index.append(synonyms)
        return SynonymsCollection(by_index, by_synonym)

    def has_synonym(self, synonym):
        assert(type(synonym) == unicode)
        synonym = env.stemmer.lemmatize_to_str(synonym)
        return synonym in self.by_synonym

    def get_synonyms(self, synonym):
        assert(type(synonym) == unicode)
        synonym = env.stemmer.lemmatize_to_str(synonym)
        index = self.by_synonym[synonym]
        return self.by_index[index]
