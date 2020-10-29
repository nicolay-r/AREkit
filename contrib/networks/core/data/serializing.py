from arekit.common.experiment.data.serializing import SerializationData


class NetworkSerializationData(SerializationData):

    def __init__(self, labels_scaler):
        super(NetworkSerializationData, self).__init__(labels_scaler=labels_scaler)

    @property
    def WordEmbedding(self):
        raise NotImplementedError()
