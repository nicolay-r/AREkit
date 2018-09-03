# -*- coding: utf-8 -*-
from pymystem3 import Mystem


# TODO. Add POS tags
class Stemmer:
    """ Yandex MyStem wrapper

        part of speech description:
        https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
    """

    pos_unknown = u"unknown"

    pos_names = [u"a", u"adv", u"advpro", u"anum", u"apro", u"com", u"conj",
                 u"intj", u"num", u"part", u"pr", u"s", u"spro", u"v", pos_unknown]

    def __init__(self):
        self.mystem = Mystem(entire_input=False)

    def lemmatize_to_list(self, text):
        return self.mystem.lemmatize(text.lower())

    def lemmatize_to_str(self, text):
        assert(isinstance(text, unicode))
        lemmas = self.mystem.lemmatize(text.lower())

        result = " ".join(lemmas)

        # print '"%s"->"%s"' % (text.encode('utf-8'), result.encode('utf-8')), ' ', len(lemmas)

        # The problem when 'G8' word, it will not be lemmatized, so next line
        # is a fix
        if len(result) == 0:
            result = text

        assert(isinstance(result, unicode))
        return result

    @staticmethod
    def _get_pos(a):
        pos = a['gr'].split(',')[0]
        if '=' in pos:
            pos = pos.split('=')[0]
        return pos

    def get_term_pos(self, term):
        assert(isinstance(term, unicode))
        analyzed = self.mystem.analyze(term)
        return self._get_term_pos(analyzed[0]) if len(analyzed) > 0 else self.pos_unknown

    def get_terms_pos(self, terms):
        """ list of part of speech according to the certain word in text
        """
        assert(isinstance(terms, list))
        pos_list = []
        for term in terms:
            analyzed = self.mystem.analyze(term)
            pos = self._get_term_pos(analyzed[0]) if len(analyzed) > 0 else self.pos_unknown
            pos_list.append(pos)

        return pos_list

    def _get_term_pos(self, analysis):
        """
        part of speech description:
            https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
        returns: str or None
        """
        if 'analysis' not in analysis:
            return self.pos_unknown

        info = analysis['analysis']
        if len(info) == 0:
            return self.pos_unknown

        return self._get_pos(info[0])

    def pos_to_int(self, pos):
        assert(isinstance(pos, unicode))
        pos = pos.lower()
        return self.pos_names.index(pos)
