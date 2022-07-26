from arekit.common.text_opinions.base import TextOpinion


class ContextOpinion(TextOpinion):
    """ This is a text opnion that is a part of a text piece, dubbed as context.
    """

    def __int__(self, doc_id, text_opinion_id, source_id, target_id, label, context_id):
        """ context_id: it might be sentence index or a combination of them in case of a complex cases.
        """
        super(ContextOpinion, self).__init__(doc_id=doc_id,
                                             text_opinion_id=text_opinion_id,
                                             source_id=source_id,
                                             target_id=target_id,
                                             label=label)

        self.__context_id = context_id

    @property
    def ContextID(self):
        return self.__context_id
