import texterra
from core.processing.lemmatization.base import Stemmer


class TexterraLemmatizationWrap(Stemmer):

    default_host = "localhost"
    default_port = "8082"

    def __init__(self, host=None, debug=False):
        default_url = 'http://{}:{}/texterra/'.format(
            self.default_host,
            self.default_port)

        url = default_url if host is None else host

        if debug:
            print "Connecting to Texterra server: {}".format(url)

        self.t = texterra.API(host=url)

    def lemmatize_to_list(self, text):
        return self._lemmatize(text)

    def lemmatize_to_str(self, text, remove_new_lines=True):
        lemmas = self._lemmatize(text)
        return " ".join(lemmas)

    def _lemmatize(self, text):
        results = self.t.lemmatization(text)
        lemmas = []
        for r in results:
            for l in r:
                i, j, original, lemma = l
                result_lemma = lemma.strip()
                lemmas.append(result_lemma if len(result_lemma) > 0 else original)
                print '"{}"'.format(lemma.encode('utf-8'))
        return lemmas
