from arekit.common.experiment.data.base import DataIO


class SerializationData(DataIO):
    """ Data, that is necessary for models training stage.
    """

    def __init__(self, labels_scale):
        super(SerializationData, self).__init__(labels_scale)

    @property
    def DistanceInTermsBetweenOpinionEndsBound(self):
        raise NotImplementedError()

    @property
    def Stemmer(self):
        raise NotImplementedError()

    @property
    def StringEntityFormatter(self):
        raise NotImplementedError()

    @property
    def FramesCollection(self):
        raise NotImplementedError()

    @property
    def FrameVariantCollection(self):
        raise NotImplementedError()

    @property
    def TermsPerContext(self):
        raise NotImplementedError
