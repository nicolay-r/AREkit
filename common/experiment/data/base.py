from arekit.common.experiment.cv.base import BaseCVFolding
from arekit.common.experiment.neutral.annot.three_scale import ThreeScaleNeutralAnnotator
from arekit.common.experiment.neutral.annot.two_scale import TwoScaleNeutralAnnotator
from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.scales.three import ThreeLabelScaler
from arekit.common.experiment.scales.two import TwoLabelScaler
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
        self.__cv_folding_algo = BaseCVFolding()
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
        return self.__model_io

    @property
    def NeutralAnnotator(self):
        """ Provides an instance of neutral annotator that might be utlized
            for neutral attitudes labeling for a specific set of documents,
            declared in a particular experiment (see OpinionOperations).
        """
        return self.__neutral_annot

    @property
    def CVFoldingAlgorithm(self):
        """ Algorithm, utilized in order to provide cross-validation split
            for experiment data-types.
        """
        return self.__cv_folding_algo

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

    def set_cv_folding_algorithm(self, cv_folding_algo):
        """ Providing cv_folding algorithm instance.
        """
        assert(isinstance(cv_folding_algo, BaseCVFolding))
        self.__cv_folding_algo = cv_folding_algo
