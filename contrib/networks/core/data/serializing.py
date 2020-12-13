from arekit.common.experiment.data.serializing import SerializationData


class NetworkSerializationData(SerializationData):

    def __init__(self, labels_scaler, stemmer):
        super(NetworkSerializationData, self).__init__(labels_scaler=labels_scaler,
                                                       stemmer=stemmer)

    @property
    def WordEmbedding(self):
        raise NotImplementedError()

    @property
    def PosTagger(self):
        raise NotImplementedError()
