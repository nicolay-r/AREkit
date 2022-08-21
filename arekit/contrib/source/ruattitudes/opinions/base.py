class SentenceOpinion(object):
    """ Provides an opinion within a sentence.
        Specific for RuAttitudes collection, as the latter provides
        connections within a sentence.
    """

    def __init__(self, source_id, target_id, label_int, tag):
        assert(isinstance(label_int, int))
        self.__label_int = label_int
        self.__source_id = source_id
        self.__target_id = target_id
        self.__tag = tag

    @property
    def SourceID(self):
        return self.__source_id

    @property
    def TargetID(self):
        return self.__target_id

    @property
    def Label(self):
        return self.__label_int

    @property
    def Tag(self):
        return self.__tag
