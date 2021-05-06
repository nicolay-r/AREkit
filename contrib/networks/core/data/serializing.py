from arekit.common.experiment.data.serializing import SerializationData
from arekit.common.experiment.input.providers.label.multiple import MultipleLabelProvider


class NetworkSerializationData(SerializationData):

    def __init__(self, labels_scaler, stemmer):
        super(NetworkSerializationData, self).__init__(label_scaler=labels_scaler, stemmer=stemmer)
        self.__label_provider = MultipleLabelProvider(labels_scaler)

    @property
    def LabelProvider(self):
        return self.__label_provider

    @property
    def FrameRolesLabelScaler(self):
        raise NotImplementedError()

    @property
    def WordEmbedding(self):
        raise NotImplementedError()

    @property
    def PosTagger(self):
        raise NotImplementedError()
