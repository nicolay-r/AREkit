from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.contrib.source.ruattitudes.news.parse_options import RuAttitudesParseOptions


class RuAttitudesDocumentOperations(DocumentOperations):

    def __init__(self, data_io):
        assert(isinstance(data_io, DataIO))
        super(RuAttitudesDocumentOperations, self).__init__()
        self.__data_io = data_io
        self.__ru_attitudes = None

    def set_ru_attitudes(self, ra):
        assert(isinstance(ra, dict))
        self.__ru_attitudes = ra

    # region DocumentOperations

    def read_news(self, doc_id):
        return self.__ru_attitudes[doc_id]

    def iter_news_indices(self, data_type):
        if data_type == DataType.Train:
            for doc_id in self.__ru_attitudes.iterkeys():
                yield doc_id

    def create_parse_options(self):
        return RuAttitudesParseOptions(stemmer=self.__data_io.Stemmer,
                                       frame_variants_collection=self.__data_io.FrameVariantCollection)

    def iter_supported_data_types(self):
        yield DataType.Train

    # endregion