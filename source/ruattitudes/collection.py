from arekit.source.ruattitudes.io_utils import RuAttitudesIOUtils, RuAttitudesVersions
from arekit.source.ruattitudes.reader import RuAttitudesFormatReader


class RuAttitudesCollection(object):

    @staticmethod
    def iter_news(version):
        """
        RuAttitudes collection reader from zip archive
        """
        assert(isinstance(version, RuAttitudesVersions))

        it = RuAttitudesIOUtils.iter_from_zip(
            inner_path=RuAttitudesIOUtils.get_collection_filepath(),
            process_func=lambda input_file: RuAttitudesFormatReader.iter_news(input_file=input_file),
            version=version)

        for news in it:
            yield news
