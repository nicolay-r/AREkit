class BaseLinkedTextOpinionsInstancesProvider(object):

    def iter_instances(self, linked_wrap):
        raise NotImplementedError()

    @staticmethod
    def provide_label(linked_wrap):
        return linked_wrap.First.Sentiment