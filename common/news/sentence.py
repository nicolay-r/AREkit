
class BaseNewsSentence(object):

    def __init__(self, text):
        assert(isinstance(text, unicode))
        self.__text = text

    @property
    def Text(self):
        return self.__text