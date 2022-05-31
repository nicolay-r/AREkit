from arekit.common.text.stemmer import Stemmer
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollectionHelper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


class RuSentRelSynonymsCollectionProvider(object):

    @staticmethod
    def load_collection(stemmer, is_read_only=True, debug=False, version=RuSentRelVersions.V11):
        assert(isinstance(stemmer, Stemmer))
        return StemmerBasedSynonymCollection(
            iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(version),
            debug=debug,
            stemmer=stemmer,
            is_read_only=is_read_only)
