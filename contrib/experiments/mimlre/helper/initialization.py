from arekit.contrib.experiments.context.helpers.initialization import ContextModelInitHelper
from arekit.networks.context.sample import InputSample
from arekit.networks.multi.training.bags import MultiInstanceBagsCollection


class MIMLREModelInitHelper(ContextModelInitHelper):

    def __init__(self, io, config):
        super(MIMLREModelInitHelper, self).__init__(io=io, config=config)

    @staticmethod
    def create_bags_collection(text_opinions_collection, frames_collection, synonyms_collection, data_type, config):
        return MultiInstanceBagsCollection.from_linked_text_opinions(
            text_opinions_collection,
            max_bag_size=config.BagSize,
            data_type=data_type,
            shuffle=True,
            create_empty_sample_func=lambda: InputSample.create_empty(config),
            create_sample_func=lambda opinion: MIMLREModelInitHelper.create_sample(
                text_opinion=opinion,
                frames_collection=frames_collection,
                synonyms_collection=synonyms_collection,
                config=config))
