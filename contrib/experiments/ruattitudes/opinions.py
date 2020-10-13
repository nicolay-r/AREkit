from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.formats.opinions import OpinionOperations
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.source.ruattitudes.news.helper import RuAttitudesNewsHelper


class RuAttitudesOpinionOperations(OpinionOperations):

    def __init__(self, data_io, experiment_name, neutral_annot_name):
        assert(isinstance(data_io, DataIO))

        # TODO: DUPLICATED WITH RuSentRel experiment.
        # TODO. Remove duplication
        neutral_root = get_path_of_subfolder_in_experiments_dir(
            experiments_dir=data_io.get_input_samples_dir(experiment_name),
            subfolder_name=neutral_annot_name)

        super(RuAttitudesOpinionOperations, self).__init__(neutral_root=neutral_root)

        self._set_synonyms_collection(data_io.SynonymsCollection)

        self.__data_io = data_io
        self.__ru_attitudes = None

    def set_ru_attitudes(self, ra):
        assert(isinstance(ra, dict))
        self.__ru_attitudes = ra

    # region private methods

    def __get_opinions_in_news(self, doc_id, opinion_check=lambda _: True):
        news = self.__ru_attitudes[doc_id]
        return [opinion
                for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)
                if opinion_check(opinion)]

    # endregion

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        return OpinionCollection.init_as_custom(opinions=self.__get_opinions_in_news(doc_id),
                                                synonyms=self.__data_io.SynonymsCollection)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, DataType))

        if data_type == DataType.Train:
            return self.__get_opinions_in_news(doc_id=doc_id,
                                               opinion_check=lambda opinion: opinion.Sentiment == NeutralLabel())
