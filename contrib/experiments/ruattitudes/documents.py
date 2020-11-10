from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.experiment.formats.documents import DocumentOperations
from arekit.contrib.source.ruattitudes.news.parse_options import RuAttitudesParseOptions


class RuAttitudesDocumentOperations(DocumentOperations):

    def __init__(self, data_io, folding, ru_attitudes):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(ru_attitudes, dict) or ru_attitudes is None)
        super(RuAttitudesDocumentOperations, self).__init__(folding)
        self.__data_io = data_io
        self.__ru_attitudes = ru_attitudes

    # region DocumentOperations

    def read_news(self, doc_id):
        return self.__ru_attitudes[doc_id]

    def _create_parse_options(self):
        assert(isinstance(self.__data_io, SerializationData))
        return RuAttitudesParseOptions(stemmer=self.__data_io.Stemmer,
                                       frame_variants_collection=self.__data_io.FrameVariantCollection)

    def iter_doc_ids_to_neutrally_annotate(self):
        return
        yield

    # endregion