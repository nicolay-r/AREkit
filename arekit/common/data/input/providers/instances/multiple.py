from arekit.common.data.input.providers.instances.base import BaseLinkedTextOpinionsInstancesProvider
from arekit.common.linked.text_opinions.wrapper import LinkedTextOpinionsWrapper
from arekit.common.text_opinions.base import TextOpinion


class MultipleLinkedTextOpinionsInstancesProvider(BaseLinkedTextOpinionsInstancesProvider):

    def __init__(self, supported_labels):
        assert(isinstance(supported_labels, list))
        self.__supported_labels = supported_labels

    def iter_instances(self, linked_wrap):
        """ Enumerate all opinions as if it would be with the different label types.
        """
        for label in self.__supported_labels:
            yield self.__modify_first_and_copy_linked_wrap(linked_wrap, label)

    @staticmethod
    def __modify_first_and_copy_linked_wrap(linked_wrap, label):
        assert (isinstance(linked_wrap, LinkedTextOpinionsWrapper))

        linked_text_opinions = [opinion for opinion in linked_wrap]
        text_opinion_copy = TextOpinion.create_copy(other=linked_text_opinions[0])
        text_opinion_copy.set_label(label=label)
        linked_text_opinions[0] = text_opinion_copy

        return LinkedTextOpinionsWrapper(linked_text_opinions=linked_text_opinions)
