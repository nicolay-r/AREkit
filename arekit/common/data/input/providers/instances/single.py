from arekit.common.data.input.providers.instances.base import BaseTextOpinionsLinkageInstancesProvider


class SingleInstanceTextOpinionsLinkageProvider(BaseTextOpinionsLinkageInstancesProvider):

    def iter_instances(self, text_opinions_linkage):
        yield text_opinions_linkage
        return