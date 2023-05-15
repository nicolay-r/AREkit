from arekit.common.data.input.providers.instances.base import BaseLinkedDataInstancesProvider
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.text_opinions.base import TextOpinion


class MultipleInstancesLinkedTextOpinionsProvider(BaseLinkedDataInstancesProvider):

    def __init__(self, supported_labels):
        assert(isinstance(supported_labels, list))
        self.__supported_labels = supported_labels

    def iter_instances(self, linked_data):
        """ Enumerate all opinions as if it would be with the different label types.
        """
        for label in self.__supported_labels:
            yield self.__modify_first_and_copy_linked_wrap(linked_data, label)

    @staticmethod
    def __modify_first_and_copy_linked_wrap(text_opinions_linkage, label):
        assert (isinstance(text_opinions_linkage, TextOpinionsLinkage))

        linkage = list(text_opinions_linkage)
        text_opinion_copy = TextOpinion.create_copy(other=linkage[0])
        text_opinion_copy.set_label(label=label)
        linkage[0] = text_opinion_copy

        return TextOpinionsLinkage(linked_data=linkage)
