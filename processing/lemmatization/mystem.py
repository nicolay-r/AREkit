from core.processing.lemmatization.base import Stemmer
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
        self.entire_input = entire_input
        self.mystem = Mystem(entire_input=entire_input)

    def lemmatize_to_list(self, text):
        return self._lemmatize_core(text)

    def lemmatize_to_str(self, text):
        result = u" ".join(self._lemmatize_core(text))

        # print '"%s"->"%s"' % (text.encode('utf-8'),
        # result.encode('utf-8')), ' ', len(lemmas)
        # The problem when 'G8' word, it will not be
        # lemmatized, so next line is a hot fix
        return result if len(result) != 0 else text.lower()

    def _lemmatize_core(self, text):
        assert(isinstance(text, unicode))
        result_list = self.mystem.lemmatize(text.lower())
        return self._filter_whitespaces(result_list)

    @staticmethod
    def _filter_whitespaces(terms):
        return [term.strip() for term in terms if term.strip()]
