import io
from core.common.synonyms import SynonymsCollection
from core.processing.lemmatization.base import Stemmer


class RuSentRelSynonymsCollection(SynonymsCollection):

    @classmethod
    def from_file(cls, filepath, stemmer, is_read_only=True, debug=False):
        assert(isinstance(filepath, unicode))
        assert(isinstance(stemmer, Stemmer))
        by_index = []
        by_synonym = {}
        cls.__from_file(filepath, by_index, by_synonym, stemmer, debug)
        return cls(by_index=by_index,
                   by_synonym=by_synonym,
                   stemmer=stemmer,
                   is_read_only=is_read_only)

    @staticmethod
    def __from_file(filepath, by_index, by_synonym, stemmer, debug):
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
