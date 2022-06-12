from arekit.common.labels.base import Label
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.base import TextOpinion


class SentenceOpinion(object):
    """
    Provides an opinion within a sentence.
    Specific for RuAttitudes collection, as the latter provides connections within a sentence.
    """

    def __init__(self, source_id, target_id, label, tag):
        assert(isinstance(source_id, int))
        assert(isinstance(target_id, int))
        assert(isinstance(label, Label))

        self.__source_id = source_id
        self.__target_id = target_id
        self.__label = label
        self.__tag = tag

    # region properties

    @property
    def SourceID(self):
        return self.__source_id

    @property
    def TargetID(self):
        return self.__target_id

    @property
    def Label(self):
        return self.__label

    @property
    def Tag(self):
        return self.__tag

    # endregion

    def to_text_opinion(self, doc_id, end_to_doc_id_func, text_opinion_id):
        """
        Converts opinion into document-level referenced opinion
        """
        return TextOpinion(doc_id=doc_id,
                           text_opinion_id=text_opinion_id,
                           source_id=end_to_doc_id_func(self.__source_id),
                           target_id=end_to_doc_id_func(self.__target_id),
                           owner=None,
                           label=self.__label)

    def to_opinion(self, source_value, target_value):
        """
        Converts onto document, non referenced opinion
        (non bounded to the text).
        """
        opinion = Opinion(source_value=source_value,
                          target_value=target_value,
                          sentiment=self.__label)

        # Using this tag allows to perform a revert operation,
        # i.e. to find opinion_ref by opinion.
        opinion.set_tag(self.__tag)

        return opinion
