import glob
import logging
import os
import shutil

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.common.model.model_io import BaseModelIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataIO(object):

    def __init__(self, labels_scale):
        assert(isinstance(labels_scale, BaseLabelScaler))
        self.__model_name = None
        self.__labels_scale = labels_scale

    # region Properties

    @property
    def LabelsScaler(self):
        return self.__labels_scale

    @property
    def SynonymsCollection(self):
        raise NotImplementedError()

    # TODO. It is both utlized on serialization stage (neut formatters), and evaluation.
    @property
    def OpinionFormatter(self):
        raise NotImplementedError()

    # region Serialization stage

    # TODO. Strongly a part of Serialization Stage.
    @property
    def DistanceInTermsBetweenOpinionEndsBound(self):
        raise NotImplementedError()

    # TODO. This should be a part of Serialization Stage.
    @property
    def Stemmer(self):
        raise NotImplementedError()

    # TODO. This should be a part of Serialization Stage.
    @property
    def StringEntityFormatter(self):
        raise NotImplementedError()

    # TODO. This should be a part of Serialization Stage.
    @property
    def FramesCollection(self):
        raise NotImplementedError()

    # TODO. This should be a part of Serialization Stage.
    @property
    def FrameVariantCollection(self):
        raise NotImplementedError()

    # TODO. This should be a part of Serialization Stage.
    @property
    def TermsPerContext(self):
        raise NotImplementedError

    def prepare_model_root(self, rm_contents=True):

        if not rm_contents:
            return

        model_io = self.ModelIO
        assert(isinstance(model_io, BaseModelIO))
        self.__rm_dir_contents(model_io.ModelRoot)

    # endregion

    @property
    def ModelIO(self):
        raise NotImplementedError()

    @property
    def Evaluator(self):
        raise NotImplementedError()

    @property
    def CVFoldingAlgorithm(self):
        raise NotImplementedError()

    # TODO. Optional and utlized in evaluation process.
    @property
    def Callback(self):
        raise NotImplementedError()

    # endregion

    # region private methods

    @staticmethod
    def __rm_dir_contents(dir_path):
        contents = glob.glob(dir_path)
        for f in contents:
            logger.info("Removing old file/dir: {}".format(f))
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f, ignore_errors=True)

    # endregion

    def get_data_root(self):
        raise NotImplementedError()

    def get_experiments_dir(self):
        raise NotImplementedError()

    def set_model_name(self, value):
        self.__model_name = value

    def get_input_samples_dir(self, experiment_name):
        assert(isinstance(experiment_name, unicode))

        is_fixed = self.CVFoldingAlgorithm.CVCount == 1
        e_name = u"{name}_{mode}_{scale}l".format(name=experiment_name,
                                                  mode=u"fixed" if is_fixed else u"cv",
                                                  scale=self.LabelsScaler.LabelsCount)

        return get_path_of_subfolder_in_experiments_dir(subfolder_name=e_name,
                                                        experiments_dir=self.get_experiments_dir())

    def get_model_root(self, experiment_name):
        assert(isinstance(experiment_name, unicode))
        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__model_name,
            experiments_dir=self.get_input_samples_dir(experiment_name))
