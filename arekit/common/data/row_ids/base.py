from arekit.common.linkage.text_opinions import TextOpinionsLinkage


class BaseIDProvider(object):
    """
    Opinion in text is a sequence of opinions in context
    o1, o2, o3, ..., on

    o1 -- first_text_opinion
    i -- index in lined (for example: i=3 => 03)

    # TODO. #376. This should be definitely refactored. This implementation
      TODO. combines opinion-based and sample-based data sources, which allows
      TODO. us to bypass such connection via external foreign keys.

      Since we are head to remove opinions, there is a need to refactor so in a
      way of an additional column that provides such information for further connection
      between rows of different storages.
    """

    SEPARATOR = '_'
    OPINION = "o{}" + SEPARATOR
    INDEX = "i{}" + SEPARATOR

    # region 'create' methods

    @staticmethod
    def create_opinion_id(text_opinions_linkage, index_in_linked):
        assert(isinstance(text_opinions_linkage, TextOpinionsLinkage))
        assert(isinstance(index_in_linked, int))

        template = ''.join([BaseIDProvider.OPINION,
                            BaseIDProvider.INDEX])

        text_opinion_id = text_opinions_linkage.First.TextOpinionID
        assert(isinstance(text_opinion_id, int))

        return template.format(text_opinion_id,
                               index_in_linked)

    @staticmethod
    def create_sample_id(linked_opinions, index_in_linked, label_scaler):
        raise NotImplementedError()

    # endregion
