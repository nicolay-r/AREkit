# -*- coding: utf-8 -*-
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
        assert(type(text) == unicode)
        lemmas = self.mystem.lemmatize(text)

        result = " ".join(lemmas)

        # print '"%s"->"%s"' % (text, result), ' ', len(lemmas)

        # The problem when 'G8' word, it will not be lemmatized, so next line
        # is a fix
        if len(result) == 0:
            result = text

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

    def analyze_pos_list(self, terms):
        """ list of part of speach according to the certain word in text
            part of speech description:
                https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
        """
        assert(type(terms) == list)
        pos = []
        for t in terms:
            a = self.mystem.analyze(t)[0]
            if 'analysis' not in a:
                pos.append(None)
                continue

            info = a['analysis']

            if len(info) == 0:
                pos.append(None)
                continue

            pos = info[0]['gr'].lower()

        return pos
