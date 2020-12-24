from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions, RuAttitudesIOUtils
from arekit.contrib.source.rusentrel.synonyms import StemmerBasedSynonymCollection
from arekit.contrib.source.rusentrel.utils import iter_synonym_groups
from arekit.processing.lemmatization.base import Stemmer


class RuAttitudesSynonymsCollectionHelper(object):

    @staticmethod
    def load_collection(stemmer, is_read_only=True, debug=False, version=RuAttitudesVersions.V11):
        assert(isinstance(stemmer, Stemmer))

        return StemmerBasedSynonymCollection(
            iter_group_values_lists=RuAttitudesSynonymsCollectionHelper.iter_groups(version),
            stemmer=stemmer,
            debug=debug,
            is_read_only=is_read_only)

    @staticmethod
    def iter_groups(version):
        it = RuAttitudesIOUtils.iter_from_zip(
            inner_path=RuAttitudesIOUtils.get_synonyms_innerpath(),
            process_func=lambda input_file: iter_synonym_groups(
                input_file,
                desc="Loading RuAttitudes SynonymsCollection"),
            version=version)

        for group in it:
            yield group