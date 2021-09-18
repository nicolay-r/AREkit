from arekit.common.labels.base import Label


class FrameState(object):

    def __init__(self, role, label, prob):
        assert(isinstance(role, str))
        assert(isinstance(label, Label))
        assert(isinstance(prob, float))
        self.__role = role
        self.__label = label
        self.__prob = prob

    @property
    def Role(self):
        return self.__role

    @property
    def Label(self):
        return self.__label

    @property
    def Prob(self):
        return self.__prob