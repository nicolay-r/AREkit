from arekit.common.experiment.data.serializing import SerializationData


class NetworkSerializationData(SerializationData):

    def __init__(self, labels_scale):
        super(NetworkSerializationData, self).__init__(labels_scale)

    @property
    def WordEmbedding(self):
        raise NotImplementedError()
