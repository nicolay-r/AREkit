from arekit.common.experiment.data_io import DataIO
from arekit.common.experiment.formats.base import BaseExperiment


class CVBasedExperiment(BaseExperiment):

    def __init__(self, data_io, prepare_model_root):
        assert(isinstance(data_io, DataIO))
        super(CVBasedExperiment, self).__init__(data_io=data_io,
                                                prepare_model_root=prepare_model_root)
