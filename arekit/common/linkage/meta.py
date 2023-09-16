from arekit.common.linkage.base import LinkedDataWrapper


class MetaEmptyLinkedDataWrapper(LinkedDataWrapper):
    """ This is a placeholder data-wrapper utilized for passing system information
        while iterating through the data pipelines.
    """

    def __init__(self, doc_id, meta_data=None):
        """ meta_data:
                optional parameter which serves any information need in further.
        """
        super(MetaEmptyLinkedDataWrapper, self).__init__([])
        self.__doc_id = doc_id
        self.__meta_data = meta_data

    @property
    def RelatedDocID(self):
        return self.__doc_id

    @property
    def MetaData(self):
        return self.__meta_data
