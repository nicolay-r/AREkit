import logging

from arekit.common.synonyms import SynonymsCollection
from arekit.common.utils import progress_bar_defined
from arekit.processing.lemmatization.base import Stemmer
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils, RuSentRelVersions

logger = logging.getLogger(__name__)


class RuSentRelSynonymsCollection:

    @staticmethod
    def load_collection(stemmer, is_read_only=True, debug=False, version=RuSentRelVersions.V11):
        assert(isinstance(stemmer, Stemmer))

        by_index = []
        by_synonym = {}
        RuSentRelIOUtils.read_from_zip(
            inner_path=RuSentRelIOUtils.get_synonyms_innerpath(),
            process_func=lambda input_file: RuSentRelSynonymsCollection._from_file(
                input_file,
                by_index=by_index,
                by_synonym=by_synonym,
                stemmer=stemmer,
                debug=debug,
                desc="Loading RuSentRel SynonymsCollection"),
            version=version)

        return SynonymsCollection(by_index=by_index,
                                  by_synonym=by_synonym,
                                  stemmer=stemmer,
                                  is_read_only=is_read_only)

    @staticmethod
    def _from_file(input_file, by_index, by_synonym, stemmer, debug, desc=""):
        assert(isinstance(by_index, list))
        assert(isinstance(by_synonym, dict))
        assert(isinstance(stemmer, Stemmer))

        lines = input_file.readlines()

        lines_it = progress_bar_defined(lines, total=len(lines), desc=desc, unit="opins")

        for line in lines_it:

            line = line.decode('utf-8')

            group_index = len(by_index)
            synonym_list = []
            args = line.split(',')

            for s in args:
                value = s.strip()
                id = SynonymsCollection.create_synonym_id(stemmer, value)

                if id in by_synonym and debug:
                    logger.info("Collection already has a value '{}'. Skipped".format(value.encode('utf-8')))
                    continue

                synonym_list.append(value)
                by_synonym[id] = group_index

            by_index.append(synonym_list)
