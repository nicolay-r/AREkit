class FrameConnotationDescriptor(object):
    """
    Polarity description between source (Agent) towards dest (Theme)
    The latter is related to the roles of frame polarity.
    """

    @property
    def Source(self):
        raise NotImplementedError()

    @property
    def Destination(self):
        raise NotImplementedError()

    @property
    def Label(self):
        raise NotImplementedError()
