from core.processing.pos.base import POSTagger
from pymystem3 import Mystem


class POSMystemWrapper(POSTagger):

    _pos_adj = u"a"
    _pos_noun = u"s"

    pos_names = [_pos_noun, u"adv", u"advpro", u"anum", u"apro", u"com", u"conj",
                 u"intj", u"num", u"part", u"pr", _pos_adj, u"spro", u"v",
                 POSTagger._pos_unknown, POSTagger._pos_empty]

    def __init__(self, mystem):
        """
        entire_input: bool
            Mystem parameter that allows to keep all information from input (true) or
            remove garbage characters
        """
        assert(isinstance(mystem, Mystem))
        self.mystem = mystem

    def get_terms_pos(self, terms):
        """ list of part of speech according to the certain word in text
        """
        assert(isinstance(terms, list))
        pos_list = []
        for term in terms:
            analyzed = self.mystem.analyze(term)
            pos = self._get_term_pos(analyzed[0]) if len(analyzed) > 0 else self._pos_unknown
            pos_list.append(pos)

        return pos_list

    def get_term_pos(self, term):
        assert(isinstance(term, unicode))
        analyzed = self.mystem.analyze(term)
        return self._get_term_pos(analyzed[0]) if len(analyzed) > 0 else self._pos_unknown

    def pos_to_int(self, pos):
        assert(isinstance(pos, unicode))
        pos = pos.lower()
        return self.pos_names.index(pos)

    def _get_term_pos(self, analysis):
        """
        part of speech description:
            https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
        returns: str or None
        """
        if 'analysis' not in analysis:
            return self._pos_unknown

        info = analysis['analysis']
        if len(info) == 0:
            return self._pos_unknown

        return self._get_pos(info[0])

    @staticmethod
    def _get_pos(a):
        pos = a['gr'].split(',')[0]
        if '=' in pos:
            pos = pos.split('=')[0]
        return pos

    def is_adjective(self, pos_type):
        assert(isinstance(pos_type, unicode))
        return pos_type.lower() == self._pos_adj

    def is_noun(self, pos_type):
        assert(isinstance(pos_type, unicode))
        return pos_type.lower() == self._pos_noun

