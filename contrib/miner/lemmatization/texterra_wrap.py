import logging
import texterra
from arekit.processing.lemmatization.base import Stemmer

logger = logging.getLogger(__name__)


class TexterraLemmatizationWrap(Stemmer):

    default_host = "localhost"
    default_port = "8082"

    def __init__(self, host=None, debug=False):
        default_url = 'http://{}:{}/texterra/'.format(self.default_host,
                                                      self.default_port)

        url = default_url if host is None else host

        if debug:
            logger.info("Connecting to Texterra server: {}".format(url))

        self.__t = texterra.API(host=url)

    # region public methods

    def lemmatize_to_list(self, text):
        return self.__lemmatize(text)

    def lemmatize_to_str(self, text, remove_new_lines=True):
        lemmas = self.__lemmatize(text)
        return " ".join(lemmas)

    def is_adjective(self, pos_type):
        raise NotImplementedError()

    def is_noun(self, pos_type):
        raise NotImplementedError()

    # endregion

    # region private methods

    def __lemmatize(self, text):
        results = self.__t.lemmatization(text)
        lemmas = []
        for r in results:
            for l in r:
                i, j, original, lemma = l
                result_lemma = lemma.strip()
                lemmas.append(result_lemma if len(result_lemma) > 0 else original)
                logger.info('"{}"'.format(lemma.encode('utf-8')))
        return lemmas

    # endregion
