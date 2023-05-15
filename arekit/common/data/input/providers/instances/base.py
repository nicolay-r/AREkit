from arekit.common.linkage.base import LinkedDataWrapper


class BaseLinkedDataInstancesProvider(object):

    def iter_instances(self, linked_data):
        raise NotImplementedError()

    @staticmethod
    def provide_label(linked_data):
        """ Implementation based on the first element of the linkage.
        """
        assert(isinstance(linked_data, LinkedDataWrapper))
        return linked_data.First.Sentiment