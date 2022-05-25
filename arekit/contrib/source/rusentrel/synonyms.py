from arekit.contrib.source.synonyms.utils import iter_synonym_groups
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils


class RuSentRelSynonymsCollectionHelper(object):

    @staticmethod
    def iter_groups(version):
        it = RuSentRelIOUtils.iter_from_zip(
            inner_path=RuSentRelIOUtils.get_synonyms_innerpath(),
            process_func=lambda input_file: iter_synonym_groups(
                input_file,
                desc="Loading RuSentRel Collection"),
            version=version)

        for group in it:
            yield group
