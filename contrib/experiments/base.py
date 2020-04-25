from os import path
from os.path import join

from arekit.contrib.experiments.data_io import DataIO
from arekit.contrib.experiments.operations.documents import DocumentOperations
from arekit.contrib.experiments.operations.opinions import OpinionOperations
from arekit.contrib.experiments.utils import get_path_of_subfolder_in_experiments_dir
from arekit.common.data_type import DataType


class BaseExperiment(OpinionOperations, DocumentOperations):

    def __init__(self, data_io, prepare_model_root):
        assert(isinstance(data_io, DataIO))
        assert(isinstance(prepare_model_root, bool))

        OpinionOperations.__init__(self)

        self.__data_io = data_io

        if prepare_model_root:
            self.DataIO.prepare_model_root()

        # Setup DataIO
        # TODO. Move into data_io
        self.__data_io.Callback.set_log_dir(log_dir=path.join(self.DataIO.get_model_root(), u"log/"))
        self.__data_io.NeutralAnnotator.initialize(experiment=self)
        self.__data_io.ModelIO.set_model_root(value=self.DataIO.get_model_root())

    # region Properties

    @property
    def DataIO(self):
        return self.__data_io

    # endregion

    # region annot output filepath

    @staticmethod
    def __data_type_to_string(data_type):
        if data_type == DataType.Train:
            return u'train'
        if data_type == DataType.Test:
            return u'test'

    def create_neutral_opinion_collection_filepath(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))

        output_dir = self.DataIO.get_experiments_dir()

        root = get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.DataIO.NeutralAnnotator.AnnotatorName,
            experiments_dir=output_dir)

        filename = u"art{doc_id}.neut.{d_type}.txt".format(
            doc_id=doc_id,
            d_type=BaseExperiment.__data_type_to_string(data_type))

        return join(root, filename)

    # endregion
