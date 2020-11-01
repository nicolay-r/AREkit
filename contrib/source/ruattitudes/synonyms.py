from arekit.common.synonyms import SynonymsCollection
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions, RuAttitudesIOUtils
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.processing.lemmatization.base import Stemmer


class RuAttitudesSynonymsCollection:

    @staticmethod
    def load_collection(stemmer, is_read_only=True, debug=False, version=RuAttitudesVersions.V11):
        assert(isinstance(stemmer, Stemmer))

        by_index = []
        by_synonym = {}
        RuAttitudesIOUtils.read_from_zip(
            inner_path=RuAttitudesIOUtils.get_synonyms_innerpath(),
            process_func=lambda input_file: RuSentRelSynonymsCollection._from_file(
                input_file,
                by_index=by_index,
                by_synonym=by_synonym,
                stemmer=stemmer,
                debug=debug,
                desc="Loading RuAttitudes SynonymsCollection"),
            version=version)

        return SynonymsCollection(by_index=by_index,
                                  by_synonym=by_synonym,
                                  stemmer=stemmer,
                                  is_read_only=is_read_only)
