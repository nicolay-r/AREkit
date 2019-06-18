from core.common.opinions.opinion import Opinion


class RuSentRelOpinion(Opinion):

    def __init__(self, value_source, value_target, sentiment):
        assert(',' not in value_source)
        assert(',' not in value_target)
        super(RuSentRelOpinion, self).__init__(value_left=value_source,
                                               value_right=value_target,
                                               sentiment=sentiment)
