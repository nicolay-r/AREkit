class FrameRole(object):

    def __init__(self, source, description):
        assert(isinstance(source, str))
        assert(isinstance(description, str))
        self.__source = source
        self.__description = description

    @property
    def Source(self):
        return self.__source

    @property
    def Description(self):
        return self.__description