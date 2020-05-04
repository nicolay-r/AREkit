import glob
import os
import shutil

from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator

from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler

from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir


class DataIO(object):

    def __init__(self, labels_scale):
        assert(isinstance(labels_scale, BaseLabelScaler))
        self.__model_name = None
        self.__labels_scale = labels_scale
        self.__neutral_annot = self.__init_annotator()

    def __init_annotator(self):
        if isinstance(self.LabelsScale, ThreeLabelScaler):
            return TwoScaleNeutralAnnotator()
        if isinstance(self.LabelsScale, TwoLabelScaler):
            return ThreeScaleNeutralAnnotator()

    # region Properties

    # TODO. To Experiment
    @property
    def NeutralAnnotator(self):
        return self.__neutral_annot

    @property
    def LabelsScale(self):
        return self.__labels_scale

    @property
    def Stemmer(self):
        raise NotImplementedError()

    @property
    def SynonymsCollection(self):
        raise NotImplementedError()

    @property
    def FramesCollection(self):
        raise NotImplementedError()

    @property
    def FrameVariantCollection(self):
        raise NotImplementedError()

    @property
    def ModelIO(self):
        raise NotImplementedError()

    @property
    def Evaluator(self):
        raise NotImplementedError()

    @property
    def CVFoldingAlgorithm(self):
        raise NotImplementedError()

    @property
    def OpinionFormatter(self):
        raise NotImplementedError()

    @property
    def Callback(self):
        raise NotImplementedError()

    # TODO. In future Proposal to move from nn configs here.
    @property
    def TermsPerContext(self):
        raise NotImplementedError

    @property
    def KeepTokens(self):
        return True

    # endregion

    # region private methods

    @staticmethod
    def __rm_dir_contents(dir_path):
        contents = glob.glob(dir_path)
        for f in contents:
            print "Removing old file/dir: {}".format(f)
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

    def get_model_root(self):
        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__model_name,
            experiments_dir=self.get_experiments_dir())

    def prepare_model_root(self, rm_contents=True):

        if not rm_contents:
            return

        self.__rm_dir_contents(self.get_model_root())

