from arekit.contrib.source.brat.relation import BratRelation


class SentenceOpinion(BratRelation):
    """ Provides an opinion within a sentence.
        Specific for RuAttitudes collection, as the latter provides
        connections within a sentence.
    """

    def __init__(self, source_id, target_id, label_int, tag):
        assert(isinstance(label_int, int))
        super(SentenceOpinion, self).__init__(source_id=source_id,
                                              target_id=target_id,
                                              rel_type="",
                                              id_in_doc="")

        self.__label_int = label_int
        self.__tag = tag

    @property
    def Label(self):
        return self.__label_int

    @property
    def Tag(self):
        return self.__tag
