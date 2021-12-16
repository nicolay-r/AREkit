from arekit.common.data.input.providers.instances.base import BaseTextOpinionsLinkageInstancesProvider
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.text_opinions.base import TextOpinion


class MultipleLinkedTextOpinionsInstancesProvider(BaseTextOpinionsLinkageInstancesProvider):

    def __init__(self, supported_labels):
        assert(isinstance(supported_labels, list))
        self.__supported_labels = supported_labels

    def iter_instances(self, text_opinion_linkage):
        """ Enumerate all opinions as if it would be with the different label types.
        """
        for label in self.__supported_labels:
            yield self.__modify_first_and_copy_linked_wrap(text_opinion_linkage, label)

    @staticmethod
    def __modify_first_and_copy_linked_wrap(text_opinions_linkage, label):
        assert (isinstance(text_opinions_linkage, TextOpinionsLinkage))

        linkage = [text_opinion for text_opinion in text_opinions_linkage]
        text_opinion_copy = TextOpinion.create_copy(other=linkage[0])
        text_opinion_copy.set_label(label=label)
        linkage[0] = text_opinion_copy

        return TextOpinionsLinkage(text_opinions_it=linkage)
