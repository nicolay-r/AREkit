# -*- coding: utf-8 -*-
from pymystem3 import Mystem


# TODO. Add POS tags
class Stemmer:
    """ MyStem wrapper
    """

    def __init__(self):
        self.mystem = Mystem(entire_input=False)

    def lemmatize_to_list(self, text):
        return self.mystem.lemmatize(text.lower())

    def lemmatize_to_str(self, text):
        assert(type(text) == unicode)
        lemmas = self.mystem.lemmatize(text.lower())

        result = " ".join(lemmas)

        # print '"%s"->"%s"' % (text.encode('utf-8'), result.encode('utf-8')), ' ', len(lemmas)

        # The problem when 'G8' word, it will not be lemmatized, so next line
        # is a fix
        if len(result) == 0:
            result = text

        assert(type(result) == unicode)
        return result

    @staticmethod
    def _get_pos(a):
        pos = a['gr'].split(',')[0]
        if '=' in pos:
            pos = pos.split('=')[0]
        return pos

    def lemmatize_to_rusvectores_str(self, text):
        """ <lemma>_<POS tag>
        """

        result = []
        analysis = self.mystem.analyze(text.lower())

        for item in analysis:
            info = item['analysis']
            if len(info) == 0:
                continue
            a = info[0]
            result.append("%s_%s" % (a['lex'], self._get_pos(a)))

        return result

    def analyze(self, text):
        """ mystem analyzer
        """
        return self.mystem.analyze(text.lower())

    def get_term_pos(self, term):
        assert(type(term) == unicode)
        return self._get_term_pos(self.mystem.analyze(term)[0])

    def get_terms_pos(self, terms):
        """ list of part of speech according to the certain word in text
        """
        assert(type(terms) == list)
        return [self._get_term_pos(self.mystem.analyze(t)[0]) for t in terms]

    def _get_term_pos(self, analysis):
        """
            part of speech description:
                https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
        """
        if 'analysis' not in analysis:
            return None

        info = analysis['analysis']
        if len(info) == 0:
            return None

        return self._get_pos(info[0])
