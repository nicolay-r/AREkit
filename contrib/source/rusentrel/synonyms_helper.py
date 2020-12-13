import logging

from arekit.contrib.source.rusentrel.synonyms import StemmerBasedSynonymCollection
from arekit.contrib.source.rusentrel.utils import iter_synonym_groups
from arekit.processing.lemmatization.base import Stemmer
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions

logger = logging.getLogger(__name__)


class RuSentRelSynonymsCollectionHelper(object):

    @staticmethod
    def load_collection(stemmer, is_read_only=True, debug=False, version=RuSentRelVersions.V11):
        assert(isinstance(stemmer, Stemmer))
        return StemmerBasedSynonymCollection(iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(version),
                                             debug=debug,
                                             stemmer=stemmer,
                                             is_read_only=is_read_only)

    @staticmethod
    def iter_groups(version=RuSentRelVersions.V11):
        it = RuSentRelIOUtils.iter_from_zip(
            inner_path=RuSentRelIOUtils.get_synonyms_innerpath(),
            process_func=lambda input_file: iter_synonym_groups(
                input_file,
                desc="Loading RuSentRel Collection"),
            version=version)

        for group in it:
            yield group
