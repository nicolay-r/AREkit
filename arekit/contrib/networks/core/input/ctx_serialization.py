from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext


class NetworkSerializationContext(ExperimentSerializationContext):

    def __init__(self, labels_scaler, annot, name_provider):
        super(NetworkSerializationContext, self).__init__(
            label_scaler=labels_scaler, annot=annot, name_provider=name_provider)
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