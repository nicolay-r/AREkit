from arekit.common.data.input.terms_mapper import OpinionContainingTextTermsMapper


class BertDefaultStringTextTermsMapper(OpinionContainingTextTermsMapper):
    """ We keep frame variant entities untouched, as specifics of the latter
        relies on hidden states of language models. By default implemenation of
        a base class assumes to provide an orginal frame variant value.
    """
    
    def __init__(self, entity_formatter, word_separator=' '):
        """ See https://github.com/nicolay-r/AREkit/issues/377
            for a greater details.
        """
        super(BertDefaultStringTextTermsMapper, self).__init__(entity_formatter)
        self.__word_separator = word_separator

    def map_entity(self, e_ind, entity):
        # guarantee that the base entity will be presented as a single term.
        entity_value = super(BertDefaultStringTextTermsMapper, self).map_entity(e_ind=e_ind, entity=entity)
        return entity_value.replace(self.__word_separator, '-')
