import collections
import os

from arekit.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.networks.data_type import DataType
from arekit.networks.network_io import NetworkIO
from arekit.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from arekit.source.rusentrel.helpers.parsed_news import RuSentRelParsedNewsHelper
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.news import RuSentRelNews
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from io_utils import create_dir_if_not_exists
from rusentrel_neutrals_io import RuSentRelNeutralIOUtils


class RuSentRelDataIO(NetworkIO):
    """
    RuSentRel dataset reader API.
    """

    model_root_template = u"./data/{}/"

    def __init__(self, synonyms, model_name=u'bert'):
        self.__synonyms = synonyms
        self.__model_name = model_name
        self.__rusentrel_news_ids_list = list(RuSentRelIOUtils.iter_collection_indices())

    # region properties

    @property
    def SynonymsCollection(self):
        return self.__synonyms

    @property
    def ModelName(self):
        return self.__model_name

    @property
    def RuSentRelNewsIDsList(self):
        return self.__rusentrel_news_ids_list

    # endregion

    def iter_test_data_indices(self):
        for doc_id in RuSentRelIOUtils.iter_test_indices():
            yield doc_id

    def iter_train_data_indices(self):
        for doc_id in RuSentRelIOUtils.iter_train_indices():
            yield doc_id

    def __get_eval_root_dir(self, data_type):
        cv_index = 0
        result_dir = os.path.join(
            self.model_root_template.format(self.__model_name),
            os.path.join(u"eval/{}/{}".format(data_type, cv_index)))

        create_dir_if_not_exists(result_dir)
        return result_dir

    def read_document(self, doc_id, keep_tokens):
        assert(isinstance(doc_id, int))
        assert(isinstance(keep_tokens, bool))

        entities = RuSentRelDocumentEntityCollection.read_collection(doc_id=doc_id,
                                                                     synonyms=self.__synonyms)

        news = RuSentRelNews.read_document(doc_id, entities)

        parsed_news = RuSentRelParsedNewsHelper.create_parsed_news(rusentrel_news_id=doc_id,
                                                                   rusentrel_news=news,
                                                                   keep_tokens=keep_tokens,
                                                                   stemmer=None)

        return news, parsed_news

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))
        return RuSentRelOpinionCollection.read_collection(doc_id=doc_id,
                                                          synonyms=self.__synonyms)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(data_type, unicode))

        filepath = RuSentRelNeutralIOUtils.get_rusentrel_neutral_opin_filepath(
            doc_id=doc_id,
            is_train=True if data_type == DataType.Train else False)

        return RuSentRelOpinionCollection.read_from_file(filepath=filepath,
                                                         synonyms=self.__synonyms)

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        assert(isinstance(data_type, unicode))
        assert(isinstance(doc_id, int))

        model_eval_root = self.__get_eval_root_dir(data_type=data_type)
        filepath = os.path.join(model_eval_root, u"{}.opin.txt".format(doc_id))
        create_dir_if_not_exists(filepath)
        return filepath

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        assert(isinstance(data_type, unicode))
        assert(isinstance(doc_ids, collections.Iterable))
        assert(isinstance(epoch_index, int))

        it = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=doc_ids,
            read_etalon_collection_func=lambda doc_id: self.read_etalon_opinion_collection(doc_id=doc_id),
            read_result_collection_func=lambda doc_id: RuSentRelOpinionCollection.read_from_file(
                filepath=self.create_result_opinion_collection_filepath(data_type=data_type,
                                                                        doc_id=doc_id,
                                                                        epoch_index=epoch_index),
                synonyms=self.__synonyms))

        for opinions_cmp in it:
            yield opinions_cmp
