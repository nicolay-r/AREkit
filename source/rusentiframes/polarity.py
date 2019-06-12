from core.evaluation.labels import Label


class FramePolarity(object):

    def __init__(self, src, dest, label, prob):
        assert(isinstance(src, unicode))
        assert(isinstance(dest, unicode))
        assert(isinstance(label, Label))
        assert(isinstance(prob, float))
        self.__src = src
        self.__dest = dest
        self.__label = label
        self.__prob = prob

    @property
    def Source(self):
        return self.__src

    @property
    def Destination(self):
        return self.__dest

    @property
    def Label(self):
        return self.__label

    @property
    def Prob(self):
        return self.__prob