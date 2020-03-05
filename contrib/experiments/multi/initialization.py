from arekit.contrib.experiments.single.initialization import SingleInstanceModelInitializer
from arekit.contrib.experiments.utils import create_input_sample
from arekit.contrib.networks.sample import InputSample
from arekit.networks.multi.training.bags import MultiInstanceBagsCollection


class MultiInstanceModelInitializer(SingleInstanceModelInitializer):

    def __init__(self, nn_io, config):
        super(MultiInstanceModelInitializer, self).__init__(nn_io=nn_io, config=config)

    @staticmethod
    def create_bags_collection(text_opinions_collection, frames_collection, synonyms_collection, data_type, config):
        return MultiInstanceBagsCollection.from_linked_text_opinions(
            text_opinions_collection,
            max_bag_size=config.BagSize,
            data_type=data_type,
            shuffle=True,
            create_empty_sample_func=lambda: InputSample.create_empty(config),
            create_sample_func=lambda opinion: create_input_sample(
                text_opinion=opinion,
                frames_collection=frames_collection,
                synonyms_collection=synonyms_collection,
                config=config))
