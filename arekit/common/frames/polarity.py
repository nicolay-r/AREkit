class FramePolarity(object):
    """
    Polarity description between source (Agent) towards dest (Theme)
    The latter are related to roles of frame polarity.
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
