from arekit.contrib.networks.core.feeding.bags.collection.multi import MultiInstanceBagsCollection
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.feeding.batch.base import MiniBatch
from arekit.contrib.networks.core.feeding.batch.multi import MultiInstanceMiniBatch


def create_batch_by_bags_group(bags_coolection_type, bags_group):
    if issubclass(bags_coolection_type, SingleBagsCollection):
        return MiniBatch(bags_group)
    if issubclass(bags_coolection_type, MultiInstanceBagsCollection):
        return MultiInstanceMiniBatch(bags_group)
