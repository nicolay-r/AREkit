from arekit.common.experiment.api.ctx_base import DataIO
from arekit.common.experiment.api.ctx_serialization import SerializationData
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.contrib.source.ruattitudes.news.parse_options import RuAttitudesParseOptions


class RuAttitudesDocumentOperations(DocumentOperations):

    def __init__(self, exp_data, folding, ru_attitudes):
        assert(isinstance(exp_data, DataIO))
        assert(isinstance(ru_attitudes, dict))
        super(RuAttitudesDocumentOperations, self).__init__(folding)
        # TODO. exp_data should be removed.
        self.__exp_data = exp_data
        self.__ru_attitudes = ru_attitudes

    # region DocumentOperations

    def get_doc(self, doc_id):
        return self.__ru_attitudes[doc_id]

    # TODO. This should be removed, since parse-options considered as a part
    # TODO. Of the text-parser instance!!!
    # TODO. Parse options should not be related to the particular collection.
    def _create_parse_options(self):
        assert(isinstance(self.__exp_data, SerializationData))
        return RuAttitudesParseOptions(stemmer=self.__exp_data.Stemmer,
                                       frame_variants_collection=self.__exp_data.FrameVariantCollection)

    def iter_doc_ids_to_annotate(self):
        return
        yield

    # endregion