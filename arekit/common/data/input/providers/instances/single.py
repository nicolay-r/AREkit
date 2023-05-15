from arekit.common.data.input.providers.instances.base import BaseLinkedDataInstancesProvider


class SingleInstanceLinkedDataProvider(BaseLinkedDataInstancesProvider):

    def iter_instances(self, linked_data):
        yield linked_data
        return