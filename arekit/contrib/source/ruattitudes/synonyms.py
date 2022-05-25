from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesIOUtils
from arekit.contrib.source.synonyms.utils import iter_synonym_groups


class RuAttitudesSynonymsCollectionHelper(object):

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
