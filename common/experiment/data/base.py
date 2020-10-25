from arekit.common.experiment.scales.base import BaseLabelScaler
from arekit.common.experiment.utils import get_path_of_subfolder_in_experiments_dir


class DataIO(object):
    """ This base class aggregates all the data necessary for
        cv-based experiment organization
        (data-serialization, training, etc.).
    """

    def __init__(self, labels_scale):
        assert(isinstance(labels_scale, BaseLabelScaler))
        self.__model_name = None
        self.__labels_scale = labels_scale

    @property
    def LabelsScaler(self):
        return self.__labels_scale

    @property
    def ModelIO(self):
        return None

    # region not implemented properties

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

    def set_model_name(self, value):
        self.__model_name = value

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

    def get_model_root(self, experiment_name):
        """ Denotes a folder of a particular model of a certain experiment.
        """
        assert(isinstance(experiment_name, unicode))
        return get_path_of_subfolder_in_experiments_dir(
            subfolder_name=self.__model_name,
            experiments_dir=self.get_input_samples_dir(experiment_name))
