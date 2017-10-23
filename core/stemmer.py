from pymystem3 import Mystem


# TODO: move into the processing section
class Stemmer:
    """ MyStem wrapper
    """

    def __init__(self):
        self.mystem = Mystem(entire_input=False)

    def lemmatize_to_list(self, text):
        return self.mystem.lemmatize(text)

    def lemmatize_to_str(self, text):
        lemmas = self.mystem.lemmatize(text)
        result = " ".join(lemmas)
        assert(type(result) == unicode)
        return result

    def lemmatize_to_rusvectores_str(self, text):
        """ <lemma>_<POS tag>
        """
        result = []
        analysis = self.mystem.analyze(text)

        for item in analysis:

            if len(item['analysis']) == 0:
                continue

            a = item['analysis'][0]
            lex = a['lex']
            pos = a['gr'].split(',')[0]

            result.append("%s_%s" % (lex, pos))

        return result

    def analyze(self, text):
        """ mystem analyzer
        """
        return self.mystem.analyze(text)
