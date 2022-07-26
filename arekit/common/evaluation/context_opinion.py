class ContextOpinion(object):
    """ This is a text opnion that is a part of a text piece, dubbed as context.
    """

    def __init__(self, doc_id, source_id, target_id, label, context_id):
        """ context_id: it might be sentence index or a combination of them in case of a complex cases.
        """
        self.__doc_id = doc_id
        self.__source_id = source_id
        self.__target_id = target_id
        self.__label = label
        self.__context_id = context_id
        self.__tag = None

    @property
    def DocId(self):
        return self.__doc_id

    @property
    def ContextId(self):
        return self.__context_id

    @property
    def SourceId(self):
        return self.__source_id

    @property
    def TargetId(self):
        return self.__target_id

    @property
    def Sentiment(self):
        return self.__label

    @property
    def Tag(self):
        return self.__tag

    def set_tag(self, tag):
        self.__tag = tag

    def set_label(self, label):
        self.__label = label
