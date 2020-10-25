from arekit.common.experiment.data.base import DataIO
from arekit.common.experiment.formats.base import BaseExperiment


class CVBasedExperiment(BaseExperiment):

    def __init__(self, data_io):
        assert(isinstance(data_io, DataIO))
        super(CVBasedExperiment, self).__init__(data_io=data_io)
