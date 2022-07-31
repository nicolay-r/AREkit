from arekit.contrib.utils.processing.languages.pos import PartOfSpeechType


class PartOfSpeechTypesService(object):

    __pos_names = {
        "S": PartOfSpeechType.NOUN,
        "ADV": PartOfSpeechType.ADV,
        "ADVPRO": PartOfSpeechType.ADVPRO,
        "ANUM": PartOfSpeechType.ANUM,
        "APRO": PartOfSpeechType.APRO,
        "COM": PartOfSpeechType.COM,
        "CONJ": PartOfSpeechType.CONJ,
        "INTJ": PartOfSpeechType.INTJ,
        "NUM": PartOfSpeechType.NUM,
        "PART": PartOfSpeechType.PART,
        "PR": PartOfSpeechType.PR,
        "A": PartOfSpeechType.ADJ,
        "SPRO": PartOfSpeechType.SPRO,
        "V": PartOfSpeechType.VERB,
        "UNKN": PartOfSpeechType.Unknown,
        "EMPTY": PartOfSpeechType.Empty}

    @staticmethod
    def iter_mystem_tags():
        for key, value in PartOfSpeechTypesService.__pos_names.items():
            yield key, value

    @staticmethod
    def get_mystem_from_string(value):
        return PartOfSpeechTypesService.__pos_names[value]

    @staticmethod
    def get_mystem_pos_count():
        return len(PartOfSpeechTypesService.__pos_names)

