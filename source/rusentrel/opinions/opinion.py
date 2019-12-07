from arekit.common.opinions.base import Opinion


class RuSentRelOpinion(Opinion):

    def __init__(self, value_source, value_target, sentiment):
        assert(',' not in value_source)
        assert(',' not in value_target)
        super(RuSentRelOpinion, self).__init__(source_value=value_source,
                                               target_value=value_target,
                                               sentiment=sentiment)
