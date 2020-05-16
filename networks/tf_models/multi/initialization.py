from arekit.networks.tf_models.single.initialization import SingleInstanceModelExperimentInitializer
from arekit.networks.training.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.sample import InputSample


class MultiInstanceModeExperimentInitializer(SingleInstanceModelExperimentInitializer):

    def __init__(self, experiment, config):
        super(MultiInstanceModeExperimentInitializer, self).__init__(
            experiment=experiment,
            config=config)

    @property
    def _BagCollectionType(self):
        return MultiInstanceBagsCollection

    def _create_empty_sample_func(self, config):
        return InputSample.create_empty(config)
