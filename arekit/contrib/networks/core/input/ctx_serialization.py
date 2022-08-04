from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider


class NetworkSerializationContext(object):

    def __init__(self, labels_scaler):
        super(NetworkSerializationContext, self).__init__()
        self.__label_provider = MultipleLabelProvider(labels_scaler)

    @property
    def LabelProvider(self):
        return self.__label_provider

    @property
    def FrameRolesLabelScaler(self):
        raise NotImplementedError()

    @property
    def FramesConnotationProvider(self):
        raise NotImplementedError()

    @property
    def PosTagger(self):
        raise NotImplementedError()