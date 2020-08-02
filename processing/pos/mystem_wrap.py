from core.processing.pos.base import POSTagger
from pymystem3 import Mystem


class POSMystemWrapper(POSTagger):

    PosAdjective = "A"
    PosNoun = "S"
    PosVerb = "V"

    pos_names = [PosNoun,
                 "ADV",
                 "ADVPRO",
                 "ANUM",
                 "APRO",
                 "COM",
                 "CONJ",
                 "INTJ",
                 "NUM",
                 "PART",
                 "PR",
                 PosAdjective,
                 "SPRO",
                 PosVerb,
                 POSTagger.Unknown,
                 POSTagger.Empty]

    def __init__(self, mystem):
        assert(isinstance(mystem, Mystem))
        self.__mystem = mystem

    def get_terms_pos(self, terms):
        """ list of part of speech according to the certain word in text
        """
        assert(isinstance(terms, list))
        pos_list = []
        for term in terms:
            analyzed = self.__mystem.analyze(term)
            pos = self.__get_term_pos(analyzed[0]) if len(analyzed) > 0 else self.Unknown
            pos_list.append(pos)

        return pos_list

    def get_term_pos(self, term):
        assert(isinstance(term, str))
        analyzed = self.__mystem.analyze(term)
        return self.__get_term_pos(analyzed[0]) if len(analyzed) > 0 else self.Unknown

    def pos_to_int(self, pos):
        assert(isinstance(pos, str))
        pos = pos.upper()
        return self.pos_names.index(pos)

    def __get_term_pos(self, analysis):
        """
        part of speech description:
            https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/
        returns: str or None
        """
        if 'analysis' not in analysis:
            return self.Unknown

        info = analysis['analysis']
        if len(info) == 0:
            return self.Unknown

        return self.__get_pos(info[0])

    @staticmethod
    def __get_pos(a):
        pos = a['gr'].split(',')[0]
        if '=' in pos:
            pos = pos.split('=')[0]
        return pos

    def is_adjective(self, pos_type):
        assert(isinstance(pos_type, str))
        return pos_type.lower() == self.PosAdjective

    def is_noun(self, pos_type):
        assert(isinstance(pos_type, str))
        return pos_type.lower() == self.PosNoun

    def is_verb(self, pos_type):
        assert(isinstance(pos_type, str))
        return pos_type.lower() == self.PosNoun

