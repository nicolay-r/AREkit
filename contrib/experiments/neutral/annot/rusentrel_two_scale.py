import collections
import logging
from os.path import join

import utils
from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.neutral.annot.base import BaseAnnotator
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir
from arekit.networks.data_type import DataType
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from arekit.source.rusentrel.opinions.opinion import RuSentRelOpinion
from arekit.source.rusentrel.opinions.serializer import RuSentRelOpinionCollectionSerializer

logger = logging.getLogger(__name__)


class RuSentRelTwoScaleNeutralAnnotator(BaseAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    For two scale classification task.
    """

    __annot_name = u"neutral_2_scale"

    def __init__(self, experiments_io, create_synonyms_collection):
        self.__experiments_io = experiments_io
        self.__synonyms = create_synonyms_collection()

    # region properties

    @property
    def AnnotationModelName(self):
        return self.__annot_name

    @property
    def SynonoymsCollection(self):
        return self.__synonyms

    @property
    def ExperimentsIO(self):
        return self.__experiments_io

    # endregion

    # region static methods

    @staticmethod
    def __iter_opinion_as_neutral(opinions):
        assert(isinstance(opinions, collections.Iterable))

        for opinion in opinions:
            yield RuSentRelOpinion(value_source=opinion.SourceValue,
                                   value_target=opinion.TargetValue,
                                   sentiment=NeutralLabel())

    @staticmethod
    def __data_type_to_string(data_type):
        if data_type == DataType.Train:
            return u'train'
        if data_type == DataType.Test:
            return u'test'

    # endregion

    def create(self, data_type):
        assert(isinstance(data_type, unicode))

        if data_type == DataType.Train:
            return

        for doc_id in RuSentRelIOUtils.iter_collection_indices():

            neutral_filepath = self.get_opin_filepath(
                doc_id=doc_id,
                data_type=data_type,
                output_dir=self.__experiments_io.get_experiments_dir())

            if utils.check_file_already_exsited(filepath=neutral_filepath, logger=logger):
                continue

            utils.notify_newfile_creation(filepath=neutral_filepath,
                                          data_type=data_type,
                                          logger=logger)

            neul_opin_iter = self.__iter_opinion_as_neutral(
                opinions=RuSentRelOpinionCollection.read_collection(doc_id=doc_id,
                                                                    synonyms=self.__synonyms))

            RuSentRelOpinionCollectionSerializer.save_to_file(
                collection=OpinionCollection(opinions=list(neul_opin_iter), synonyms=self.__synonyms),
                filepath=neutral_filepath)

    def get_opin_filepath(self, doc_id, data_type, output_dir):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))
        assert(isinstance(output_dir, unicode))

        root = get_path_of_subfolder_in_experiments_dir(subfolder_name=self.AnnotationModelName,
                                                        experiments_dir=output_dir)

        filename = u"art{doc_id}.neut.{d_type}.txt".format(
            doc_id=doc_id,
            d_type=RuSentRelTwoScaleNeutralAnnotator.__data_type_to_string(data_type))

        return join(root, filename)
