from arekit.common.languages.pos import PartOfSpeechType


class PartOfSpeechTypesService(object):

    __pos_names = {
        u"S": PartOfSpeechType.NOUN,
        u"ADV": PartOfSpeechType.ADV,
        u"ADVPRO": PartOfSpeechType.ADVPRO,
        u"ANUM": PartOfSpeechType.ANUM,
        u"APRO": PartOfSpeechType.APRO,
        u"COM": PartOfSpeechType.COM,
        u"CONJ": PartOfSpeechType.CONJ,
        u"INTJ": PartOfSpeechType.INTJ,
        u"NUM": PartOfSpeechType.NUM,
        u"PART": PartOfSpeechType.PART,
        u"PR": PartOfSpeechType.PR,
        u"A": PartOfSpeechType.ADJ,
        u"SPRO": PartOfSpeechType.SPRO,
        u"V": PartOfSpeechType.VERB}

    @staticmethod
    def iter_mystem_tags():
        for key, value in PartOfSpeechTypesService.__pos_names.iteritems():
            yield key, value

    @staticmethod
    def get_mystem_from_string(value):
        return PartOfSpeechTypesService.__pos_names[value]

    @staticmethod
    def get_mystem_pos_count():
        return len(PartOfSpeechTypesService.__pos_names)

