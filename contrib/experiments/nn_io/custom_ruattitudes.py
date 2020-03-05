from arekit.contrib.experiments.nn_io.cv_based import CVBasedNeuralNetworkIO
from arekit.contrib.experiments.nn_io.utils import read_ruattitudes_in_memory
from arekit.processing.lemmatization.base import Stemmer
from arekit.source.ruattitudes.helpers.parsed_news import RuAttitudesParsedNewsHelper
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection


class CustomRuAttitudesFormatIO(CVBasedNeuralNetworkIO):

    def __init__(self, model_name, cv_count, experiments_io, doc_ids):
        """
        doc_ids: set
            set of doc_ids which is supposed to saved during reading process
        """
        assert(isinstance(doc_ids, set) or doc_ids is None)
        super(CustomRuAttitudesFormatIO, self).__init__(
            model_name=model_name,
            cv_count=cv_count,
            experiments_io=experiments_io)

        self.__ra_format_docs = None

        # self.__neutral_annot = DefaultNeutralAnnotationAlgorithm(
        #     synonyms=None,
        #     create_opinion_func=None,
        #     create_opinion_collection_func=None)

    def init_synonyms_collection(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        super(CustomRuAttitudesFormatIO, self).init_synonyms_collection(stemmer)
        self.__ra_format_docs = read_ruattitudes_in_memory(stemmer)

    def read_synonyms_collection(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        super(CustomRuAttitudesFormatIO, self).read_synonyms_collection(stemmer)
        return RuSentRelSynonymsCollection.read_collection(stemmer=stemmer,
                                                           is_read_only=True)

    def read_parsed_news(self, doc_id, keep_tokens, stemmer):

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
