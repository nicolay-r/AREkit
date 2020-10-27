from arekit.common.experiment.formats.base import BaseExperiment


class CVBasedExperiment(BaseExperiment):

    def __init__(self, data_io, experiment_io):
        super(CVBasedExperiment, self).__init__(data_io=data_io,
                                                experiment_io=experiment_io)
