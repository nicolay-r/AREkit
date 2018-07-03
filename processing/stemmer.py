# -*- coding: utf-8 -*-
from pymystem3 import Mystem


# TODO. Add POS tags
class Stemmer:
    """ Yandex MyStem wrapper

        part of speech description:
        https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
    """

    pos_names = ["a", "adv", "advpro", "anum", "apro", "com", "conj",
                 "intj", "num", "part", "pr", "s", "spro", "v"]

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
        returns: str or None
        """
        if 'analysis' not in analysis:
            return None

        info = analysis['analysis']
        if len(info) == 0:
            return None

        return self._get_pos(info[0])

    def pos_to_int(self, pos):
        assert(isinstance(pos, str))
        pos = pos.lower()
        if pos in self.pos_names:
            self.pos_names.index(pos)
        return -1
