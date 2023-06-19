
class BaseDocumentSentence(object):

    def __init__(self, text):
        self.__text = text

    @property
    def Text(self):
        """
        Any type, i.e.
            - str: original text as string
            - list of words: separated by words/tokens
        """
        return self.__text
