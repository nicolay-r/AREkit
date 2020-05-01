from arekit.common.experiment.cv_based import CVBasedExperiment
from arekit.source.ruattitudes.helpers.parsed_news import RuAttitudesParsedNewsHelper

# TODO. Remove.
class CustomRuAttitudesFormatIO(CVBasedExperiment):

    def __init__(self, data_io, doc_ids):
        """
        doc_ids: set
            set of doc_ids which is supposed to saved during reading process
        """
        assert(isinstance(doc_ids, set) or doc_ids is None)
        super(CustomRuAttitudesFormatIO, self).__init__(
            data_io=data_io,
            prepare_model_root=True)

        self.__ra_format_docs = None

        # self.__neutral_annot = DefaultNeutralAnnotationAlgorithm(
        #     synonyms=None,
        #     create_opinion_func=None,
        #     create_opinion_collection_func=None)

    def read_parsed_news(self, doc_id):

        news = self.__ra_format_docs[doc_id]
        parsed_news = RuAttitudesParsedNewsHelper.create_parsed_news(doc_id=doc_id,
                                                                     news=news)

        return news, parsed_news

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))

        # TODO. Utilize default neutral annotator, based on readed document by doc_id
        raise NotImplementedError()

    def iter_train_data_indices(self):
        for doc_id in self.__ra_format_docs.iterkeys():
            yield doc_id

