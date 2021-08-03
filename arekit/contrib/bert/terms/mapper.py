from arekit.common.experiment.input.terms_mapper import OpinionContainingTextTermsMapper


class BertDefaultStringTextTermsMapper(OpinionContainingTextTermsMapper):
    """ We keep frame variant entities untouched, as specifics of the latter
        relies on hidden states of language models. By default implemenation of
        a base class assumes to provide an orginal frame variant value.
    """
    pass
