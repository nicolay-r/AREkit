from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.api.ctx_serialization import SerializationData


class NetworkSerializationData(SerializationData):

    def __init__(self, labels_scaler, annot, stemmer):
        super(NetworkSerializationData, self).__init__(
            label_scaler=labels_scaler,
            annot=annot,
            stemmer=stemmer)
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

    @property
    def StringEntityEmbeddingFormatter(self):
        raise NotImplementedError()
