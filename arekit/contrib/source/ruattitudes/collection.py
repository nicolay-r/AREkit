from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions, RuAttitudesIOUtils
from arekit.contrib.source.ruattitudes.reader import RuAttitudesFormatReader


class RuAttitudesCollection(object):

    @staticmethod
    def __get_reading_handler(input_file, read_inds_only, get_news_inds_func):
        assert(isinstance(read_inds_only, bool))

        if read_inds_only:
            return RuAttitudesFormatReader.iter_news_inds(input_file=input_file,
                                                          get_news_index_func=get_news_inds_func)
        else:
            return RuAttitudesFormatReader.iter_news(input_file=input_file,
                                                     get_news_index_func=get_news_inds_func)

    @staticmethod
    def iter_news(version, get_news_index_func, return_inds_only):
        """
        RuAttitudes collection reader from zip archive
        """
        assert(isinstance(version, RuAttitudesVersions))
        assert(callable(get_news_index_func))
        assert(isinstance(return_inds_only, bool))

        it = RuAttitudesIOUtils.iter_from_zip(
            inner_path=RuAttitudesIOUtils.get_collection_filepath(),
            process_func=lambda input_filepath: RuAttitudesCollection.__get_reading_handler(
                input_file=input_filepath,
                read_inds_only=return_inds_only,
                get_news_inds_func=get_news_index_func),
            version=version)

        for news in it:
            yield news
