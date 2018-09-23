# -*- coding: utf-8 -*-
from pymystem3 import Mystem


# TODO. Add POS tags
class Stemmer:
    """ Yandex MyStem wrapper

        part of speech description:
        https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
    """

    pos_unknown = u"unknown"
    pos_empty = u"empty"

    pos_names = [u"a", u"adv", u"advpro", u"anum", u"apro", u"com", u"conj",
                 u"intj", u"num", u"part", u"pr", u"s", u"spro", u"v",
                 pos_unknown, pos_empty]

    def __init__(self, entire_input=False):
        """
        entire_input: bool
            Mystem parameter that allows to keep all information from input (true) or
            remove garbage characters
        """
        self.entire_input = entire_input
        self.mystem = Mystem(entire_input=entire_input)

    def lemmatize_to_list(self, text):
        result_list = self.mystem.lemmatize(text.lower())

        if self.entire_input:
            return [term.strip() for term in result_list if term.strip()]

        return result_list

    def lemmatize_to_str(self, text, remove_new_lines=True):
        """
        returns: unicode
            lemmatized or original string (in case of empty lemmatization result)
        """
        assert(isinstance(text, unicode))
        lemmas = self.mystem.lemmatize(text.lower())

        if remove_new_lines:
            self._remove_new_lines(lemmas)

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

    @staticmethod
    def _remove_new_lines(terms):
        """
        Remove newline character in place
        """
        new_line = '\n'
        while new_line in terms:
            terms.remove(new_line)

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
