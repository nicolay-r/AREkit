from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.enum_input_types import ModelInputType


def create_bags_collection_type(model_input_type):
    assert(isinstance(model_input_type, ModelInputType))

    if model_input_type == ModelInputType.SingleInstance:
        return SingleBagsCollection
    if model_input_type == ModelInputType.MultiInstanceMaxPooling:
        return MultiInstanceBagsCollection
    if model_input_type == ModelInputType.MultiInstanceWithSelfAttention:
        return MultiInstanceBagsCollection