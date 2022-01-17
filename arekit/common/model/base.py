from arekit.common.experiment.data_type import DataType


class BaseModel(object):
    """ Base Model
    """

    def fit(self, model_params, seed):
        raise NotImplementedError()

    def predict(self, data_type=DataType.Test):
        raise NotImplementedError()
