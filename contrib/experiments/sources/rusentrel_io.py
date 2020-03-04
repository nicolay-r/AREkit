import collections

from arekit.contrib.experiments.doc_stat.rusentrel import RuSentRelDocStatGenerator
from arekit.contrib.experiments.sources.cv_based_io import CVBasedIO
from arekit.contrib.experiments.neutral.annot.rusentrel import RuSentRelNeutralAnnotator
from arekit.networks.data_type import DataType
from arekit.source.rusentrel.helpers.parsed_news import RuSentRelParsedNewsHelper
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from arekit.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.processing.lemmatization.base import Stemmer


class RuSentRelBasedExperimentIO(CVBasedIO):
    """
    Represents Input interface for NeuralNetwork ctx
    Now exploited (treated) as input interface only
    """

    def __init__(self, model_name, experiments_io, cv_count=1):
        super(RuSentRelBasedExperimentIO, self).__init__(
            experiments_io=experiments_io,
            cv_count=cv_count,
            model_name=model_name)

        self.__model_name = model_name
        self.__rusentrel_news_ids_list = list(RuSentRelIOUtils.iter_collection_indices())
        self.__rusentrel_news_ids = set(self.__rusentrel_news_ids_list)

        # Keys
        self.__eval_on_rusentrel_docs_key = False

    # region properties

    @property
    def RuSentRelNewsIDsList(self):
        return self.__rusentrel_news_ids_list

    @property
    def CVCount(self):
        return self.__cv_count

    @property
    def EvalOnRuSentRelDocsOnly(self):
        return self.__eval_on_rusentrel_docs_key

    # endregion

    def create_docs_stat_generator(self):
        return RuSentRelDocStatGenerator(synonyms=self.SynonymsCollection)

    def set_eval_on_rusentrel_docs_key(self, value):
        assert(isinstance(value, bool))
        self.__eval_on_rusentrel_docs_key = value

    def is_rusentrel_news_id(self, news_id):
        assert(isinstance(news_id, int))
        return news_id in self.__rusentrel_news_ids

    # region 'read' public methods

    def read_parsed_news(self, doc_id, keep_tokens, stemmer):
        assert(isinstance(doc_id, int))
        assert(isinstance(keep_tokens, bool))
        assert(isinstance(stemmer, Stemmer))

        entities = RuSentRelDocumentEntityCollection.read_collection(doc_id=doc_id,
                                                                     synonyms=self.SynonymsCollection)

        news = RuSentRelNews.read_document(doc_id, entities)

        parsed_news = RuSentRelParsedNewsHelper.create_parsed_news(rusentrel_news_id=doc_id,
                                                                   rusentrel_news=news,
                                                                   keep_tokens=keep_tokens,
                                                                   stemmer=stemmer)

        return news, parsed_news

    def read_synonyms_collection(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        return RuSentRelSynonymsCollection.read_collection(stemmer=stemmer,
                                                           is_read_only=True)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(data_type, unicode))

        # TODO. Create non-static
        filepath = RuSentRelNeutralAnnotator.get_opin_filepath(
            doc_id=doc_id,
            is_train=True if data_type == DataType.Train else False,
            experiments_io=self.__experiments_io)

        return RuSentRelOpinionCollection.read_from_file(filepath=filepath,
                                                         synonyms=self.SynonymsCollection)

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        return RuSentRelOpinionCollection.read_collection(doc_id=doc_id,
                                                          synonyms=self.SynonymsCollection)

    # endregion

    # region 'iters' public methods

    def iter_data_indices(self, data_type):
        if data_type == DataType.Train:
            return self.iter_train_data_indices()
        if data_type == DataType.Test:
            return self.iter_test_data_indices()

    def iter_test_data_indices(self):
        if self.__cv_count == 1:
            for doc_id in RuSentRelIOUtils.iter_test_indices():
                yield doc_id
        else:
            for doc_id in super(RuSentRelBasedExperimentIO, self).iter_test_data_indices():
                yield doc_id

    def iter_train_data_indices(self):
        if self.__cv_count == 1:
            for doc_id in RuSentRelIOUtils.iter_train_indices():
                yield doc_id
        else:
            for doc_id in super(RuSentRelBasedExperimentIO, self).iter_train_data_indices():
                yield doc_id

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        assert(isinstance(data_type, unicode))
        assert(isinstance(doc_ids, collections.Iterable))
        assert(isinstance(epoch_index, int))

        if self.__eval_on_rusentrel_docs_key:
            doc_ids = [doc_id for doc_id in doc_ids if doc_id in self.__rusentrel_news_ids]

        it = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=doc_ids,
            read_etalon_collection_func=lambda doc_id: self.read_etalon_opinion_collection(doc_id=doc_id),
            read_result_collection_func=lambda doc_id: RuSentRelOpinionCollection.read_from_file(
                filepath=self.create_result_opinion_collection_filepath(data_type=data_type,
                                                                        doc_id=doc_id,
                                                                        epoch_index=epoch_index),
                synonyms=self.SynonymsCollection))

        for opinions_cmp in it:
            yield opinions_cmp

    # endregion

    # region 'create' public methods

    def create_opinion_collection(self):
        return RuSentRelOpinionCollection([], synonyms=self.SynonymsCollection)

    # endregion
