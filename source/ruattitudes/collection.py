from arekit.source.ruattitudes.io_utils import RuAttitudesIOUtils, RuAttitudesVersions
from arekit.source.ruattitudes.reader import RuAttitudesFormatReader


class RuAttitudesCollection(object):

    @staticmethod
    def iter_news(get_news_index_func, stemmer=None, version=RuAttitudesVersions.V11):
        """
        RuAttitudes collection reader from zip archive
        """
        assert(callable(get_news_index_func))

        it = RuAttitudesIOUtils.iter_from_zip(
            inner_path=RuAttitudesIOUtils.get_collection_filepath(),
            process_func=lambda input_file: RuAttitudesFormatReader.iter_news(
                input_file=input_file,
                get_news_index_func=get_news_index_func,
                stemmer=stemmer),
            version=version)

        for news in it:
            yield news
