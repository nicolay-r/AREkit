from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions, RuAttitudesIOUtils
from arekit.contrib.source.ruattitudes.reader import RuAttitudesFormatReader


class RuAttitudesCollection(object):

    @staticmethod
    def iter_news(version, get_news_index_func):
        """
        RuAttitudes collection reader from zip archive
        """
        assert(isinstance(version, RuAttitudesVersions))
        assert(callable(get_news_index_func))

        it = RuAttitudesIOUtils.iter_from_zip(
            inner_path=RuAttitudesIOUtils.get_collection_filepath(),
            process_func=lambda input_file: RuAttitudesFormatReader.iter_news(
                input_file=input_file,
                get_news_index_func=get_news_index_func),
            version=version)

        for news in it:
            yield news
