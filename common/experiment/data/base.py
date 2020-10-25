from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir
from arekit.common.model.model_io import BaseModelIO


class DataIO(object):
    """ This base class aggregates all the data necessary for
        cv-based experiment organization
        (data-serialization, training, etc.).
    """

    def __init__(self, labels_scaler):
        assert(isinstance(labels_scaler, BaseLabelScaler))
        self.__labels_scale = labels_scaler
        self.__neutral_annot = self.__init_annotator(labels_scaler)
        self.__model_io = None

    @property
    def LabelsScaler(self):
        """ Declares the amount of labels utilized in experiment. The latter
            is necessary for conversions from int (uint) to Labels and vice versa.
        """
        return self.__labels_scale

    @property
    def ModelIO(self):
        """ Provides model paths for the resources utilized during training process.
            The latter is important in Neural Network training process, when there is
            a need to obtain model root directory.
        """
        return None

    @property
    def NeutralAnnotator(self):
        """ Provides an instance of neutral annotator that might be utlized
            for neutral attitudes labeling for a specific set of documents,
            declared in a particular experiment (see OpinionOperations).
        """
        return self.__neutral_annot

    # region not implemented properties

    @property
    def DistanceInTermsBetweenOpinionEndsBound(self):
        raise NotImplementedError()

    @property
    def SynonymsCollection(self):
        raise NotImplementedError()

    @property
    def OpinionFormatter(self):
        """ Corresponds to `OpinionCollectionsFormatter` instance
        """
        raise NotImplementedError()

    @property
    def CVFoldingAlgorithm(self):
        raise NotImplementedError()

    # endregion

    # region private methods

    def __init_annotator(self, label_scaler):
        if isinstance(label_scaler, TwoLabelScaler):
            return TwoScaleNeutralAnnotator()
        elif isinstance(label_scaler, ThreeLabelScaler):
            return ThreeScaleNeutralAnnotator(self.DistanceInTermsBetweenOpinionEndsBound)
        raise NotImplementedError(u"Could not create neutral annotator for scaler '{}'".format(label_scaler))

    # endregion

    def set_model_io(self, model_io):
        """ Providing model_io in experiment data.
        """
        assert(isinstance(model_io, BaseModelIO))
        self.__model_io = model_io

    def get_experiment_sources_dir(self):
        """ Provides directory for samples.
        """
        raise NotImplementedError()

    def get_input_samples_dir(self, experiment_name):
        """ Provides directory with serialized input data (samples).
            The path may vary and depends on CVFolding format, i.e.
                1 -- fixed,
                >= 1 => cv-based
        """
        assert(isinstance(experiment_name, unicode))

        is_fixed = self.CVFoldingAlgorithm.CVCount == 1
        e_name = u"{name}_{mode}_{scale}l".format(name=experiment_name,
                                                  mode=u"fixed" if is_fixed else u"cv",
                                                  scale=self.LabelsScaler.LabelsCount)

        return get_path_of_subfolder_in_experiments_dir(subfolder_name=e_name,
                                                        experiments_dir=self.get_experiment_sources_dir())
