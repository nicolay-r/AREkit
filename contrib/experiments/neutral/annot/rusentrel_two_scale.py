import logging
from os.path import join

import utils
from arekit.common.labels.base import NeutralLabel
from arekit.common.opinions.base import Opinion
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.experiments.data_io import DataIO
from arekit.contrib.experiments.neutral.annot.base import BaseAnnotator
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir
from arekit.networks.data_type import DataType
from arekit.source.rusentrel.io_utils import RuSentRelIOUtils
from arekit.source.rusentrel.opinions.collection import RuSentRelOpinionCollection

logger = logging.getLogger(__name__)


class RuSentRelTwoScaleNeutralAnnotator(BaseAnnotator):
    """
    Neutral Annotator for RuSentRel Collection (of each data_type)

    For two scale classification task.
    """

    __annot_name = u"neutral_2_scale"

    def __init__(self, data_io):
        assert(isinstance(data_io, DataIO))
        self.__data_io = data_io

    # region properties

    @property
    def AnnotationModelName(self):
        return self.__annot_name

    @property
    def DataIO(self):
        return self.__data_io

    # endregion

    # region static methods

    @staticmethod
    def __iter_opinion_as_neutral(collection):
        assert(isinstance(collection, OpinionCollection))

        for opinion in collection:
            yield Opinion(source_value=opinion.SourceValue,
                          target_value=opinion.TargetValue,
                          sentiment=NeutralLabel())

    @staticmethod
    def __data_type_to_string(data_type):
        if data_type == DataType.Train:
            return u'train'
        if data_type == DataType.Test:
            return u'test'

    def __create_opinions_for_classification(self, doc_id):
        # TODO. This should be based on samples.
        # TODO. Use helper to obtain such samples.
        # TODO. And sentence-based document processing.
        collection = RuSentRelOpinionCollection.load_collection(
            doc_id=doc_id,
            synonyms=self.__data_io.SynonymsCollection)

        neul_opin_iter = self.__iter_opinion_as_neutral(collection)

        return OpinionCollection(opinions=list(neul_opin_iter),
                                 synonyms=self.__data_io.SynonymsCollection)

    # endregion

    def _filter_non_created_doc_ids(self, all_doc_ids, data_type):

        for doc_id in all_doc_ids:

            filepath = self.get_opin_filepath(
                doc_id=doc_id,
                data_type=data_type,
                output_dir=self.__data_io.get_experiments_dir())

            if utils.check_file_already_existed(filepath=filepath, logger=logger):
                continue

            yield doc_id, filepath

    def create(self, data_type):
        assert(isinstance(data_type, unicode))

        if data_type == DataType.Train:
            return

        filtered_iter = self._filter_non_created_doc_ids(
            all_doc_ids=RuSentRelIOUtils.iter_collection_indices(),
            data_type=data_type)

        for doc_id, filepath in filtered_iter:

            utils.notify_newfile_creation(filepath=filepath,
                                          data_type=data_type,
                                          logger=logger)

            self.__data_io.OpinionFormatter.save_to_file(
                collection=self.__create_opinions_for_classification(doc_id),
                filepath=filepath)

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
