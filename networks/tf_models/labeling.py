from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.labeling import LabeledCollection


class BaseSamplesLabeling(object):

    def __init__(self, data_type, samples_labeling_collection):
        assert(isinstance(data_type, DataType))
        assert(isinstance(samples_labeling_collection, LabeledCollection))
        self.__data_type = data_type
        self.__labeling_collection = samples_labeling_collection

    def before_labeling(self):
        return len(self.__labeling_collection._labels_defined) == 0

    def predict(self, labeling_callback):
        assert(callable(labeling_callback))

        self.__labeling_collection.reset_labels()

        self.before_labeling()
        callback_output = labeling_callback()

        self.__labeling_collection.reset_labels()

        return callback_output
