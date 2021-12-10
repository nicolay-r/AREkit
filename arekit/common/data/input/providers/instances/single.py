from arekit.common.data.input.providers.instances.base import BaseLinkedTextOpinionsInstancesProvider


class SingleLinkedTextOpinionsInstancesProvider(BaseLinkedTextOpinionsInstancesProvider):

    def iter_instances(self, wrapper):
        yield wrapper
        return