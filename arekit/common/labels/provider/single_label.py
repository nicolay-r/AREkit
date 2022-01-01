from arekit.common.labels.provider.base import BasePairLabelProvider


class PairSingleLabelProvider(BasePairLabelProvider):
    """ Providing a predefined label instance irrespective
        from the source and target entity parameters.
    """

    def __init__(self, label_instance):
        super(PairSingleLabelProvider, self).__init__()
        self.__label_instance = label_instance

    def provide(self, source, target):
        return self.__label_instance
