class BaseTextOpinionsLinkageInstancesProvider(object):

    def iter_instances(self, text_opinion_linkage):
        raise NotImplementedError()

    @staticmethod
    def provide_label(text_opinion_linkage):
        return text_opinion_linkage.First.Sentiment