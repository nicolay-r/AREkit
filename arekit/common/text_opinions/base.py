from arekit.common.labels.base import Label


class TextOpinion(object):
    """
    Represents a relation which were found in news article
    and composed between two named entities
        (it was found especially by Opinion with predefined label)
        allows to modify label using set_label
    """

    # region constructors

    def __init__(self, doc_id, text_opinion_id, source_id, target_id, owner, label):
        assert(isinstance(doc_id, int))
        assert(isinstance(text_opinion_id, int) or text_opinion_id is None)

        self.__source_id = source_id
        self.__target_id = target_id
        self.__doc_id = doc_id
        self.__owner = owner
        self.__text_opinion_id = text_opinion_id
        self.__modifiable_label = label

    @classmethod
    def create_copy(cls, other, keep_text_opinion_id=True):
        assert(isinstance(other, TextOpinion))
        assert(isinstance(keep_text_opinion_id, bool))
        return cls.__try_create_copy_core(other=other, keep_text_opinion_id=keep_text_opinion_id)

    @staticmethod
    def try_convert(other, convert_func):
        """ Creates a copy of `other` opinion with different id of opinion participants.
            Use cases: required for BaseParsedNewsServiceProvider, when we decided to bring the outside
            opinion into one which is based on DocumentEntities.
        """
        assert(isinstance(other, TextOpinion))
        assert(callable(convert_func))
        return TextOpinion.__try_create_copy_core(other=other, convert_id_func=convert_func)

    @staticmethod
    def __try_create_copy_core(other, convert_id_func=lambda part_id: part_id, keep_text_opinion_id=True):
        """ Tries to compose a copy by considering an optional id conversion,
            and identification keeping.
            convert_id:
                func(id) -> id
        """
        assert(callable(convert_id_func))

        source_id = convert_id_func(other.SourceId)
        target_id = convert_id_func(other.TargetId)

        if source_id is None or target_id is None:
            return None

        return TextOpinion(doc_id=other.__doc_id,
                           text_opinion_id=other.__text_opinion_id if keep_text_opinion_id else None,
                           source_id=source_id,
                           target_id=target_id,
                           owner=other.Owner,
                           label=other.Sentiment)

    # endregion

    # region properties

    @property
    def Sentiment(self):
        return self.__modifiable_label

    @property
    def DocID(self):
        return self.__doc_id

    @property
    def TextOpinionID(self):
        return self.__text_opinion_id

    @property
    def SourceId(self):
        return self.__source_id

    @property
    def TargetId(self):
        return self.__target_id

    @property
    def Owner(self):
        return self.__owner

    # endregion

    # region public methods

    def set_text_opinion_id(self, text_opinion_id):
        assert(self.__text_opinion_id is None)
        assert(isinstance(text_opinion_id, int))
        self.__text_opinion_id = text_opinion_id

    def set_label(self, label):
        assert(isinstance(label, Label))
        self.__modifiable_label = label

    def set_owner(self, owner):
        assert(owner is not None)
        assert(self.__owner is None)
        self.__owner = owner

    # endregion
