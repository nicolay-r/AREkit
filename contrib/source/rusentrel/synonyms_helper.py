import logging

from arekit.common.utils import progress_bar_defined
from arekit.contrib.source.rusentrel.synonyms import StemmerBasedSynonymCollection
from arekit.processing.lemmatization.base import Stemmer
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions

logger = logging.getLogger(__name__)


class RuSentRelSynonymsCollectionHelper(object):

    @staticmethod
    def load_collection(stemmer, is_read_only=True, debug=False, version=RuSentRelVersions.V11):
        assert(isinstance(stemmer, Stemmer))

        it = RuSentRelIOUtils.iter_from_zip(
            inner_path=RuSentRelIOUtils.get_synonyms_innerpath(),
            process_func=lambda input_file: RuSentRelSynonymsCollectionHelper.iter_synonym_groups(
                input_file,
                desc="Loading RuSentRel Collection"),
            version=version)

        return StemmerBasedSynonymCollection(iter_group_values_lists=it,
                                             debug=debug,
                                             stemmer=stemmer,
                                             is_read_only=is_read_only)

    @staticmethod
    def iter_synonym_groups(input_file, desc=""):
        """ All the synonyms groups organized in lines, where synonyms demarcated by ',' sign
        """
        lines = input_file.readlines()

        lines_it = progress_bar_defined(lines,
                                        total=len(lines),
                                        desc=desc,
                                        unit="opins")

        for line in lines_it:
            yield line.decode('utf-8').split(',')
