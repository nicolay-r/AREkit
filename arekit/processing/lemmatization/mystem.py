from arekit.common.text.stemmer import Stemmer
from arekit.common.utils import filter_whitespaces
from pymystem3 import Mystem


class MystemWrapper(Stemmer):
    """ Yandex MyStem wrapper

        part of speech description:
        https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
    """

    def __init__(self, entire_input=False):
        """
        entire_input: bool
            Mystem parameter that allows to keep all information from input (true) or
            remove garbage characters
        """
        self.__mystem = Mystem(entire_input=entire_input)

    # region properties

    @property
    def MystemInstance(self):
        return self.__mystem

    # endregion

    # region public methods

    def lemmatize_to_list(self, text):
        return self.__lemmatize_core(text)

    def lemmatize_to_str(self, text):
        result = " ".join(self.__lemmatize_core(text))
        return result if len(result) != 0 else self.__process_original_text(text)

    # endregion

    # region private methods

    def __lemmatize_core(self, text):
        assert(isinstance(text, str))
        result_list = self.__mystem.lemmatize(self.__process_original_text(text))
        return filter_whitespaces(result_list)

    @staticmethod
    def __process_original_text(text):
        return text.lower()

    # endregion
