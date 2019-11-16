from core.common.frames.polarity import FramePolarity
from core.common.labels.base import Label


class RuSentiFramesFramePolarity(FramePolarity):
    """
    Polarity description between source (Agent) towards dest (Theme)
    The latter are related to roles of frame polarity.
    """

    def __init__(self, role_src, role_dest, label, prob):
        assert(isinstance(role_src, unicode))
        assert(isinstance(role_dest, unicode))
        assert(isinstance(label, Label))
        assert(isinstance(prob, float))
        self.__role_src = role_src
        self.__role_dest = role_dest
        self.__label = label
        self.__prob = prob

    @property
    def Source(self):
        return self.__role_src

    @property
    def Destination(self):
        return self.__role_dest

    @property
    def Label(self):
        return self.__label

    @property
    def Prob(self):
        return self.__prob
