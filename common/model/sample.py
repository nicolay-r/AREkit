from collections import OrderedDict

from arekit.common.dataset.text_opinions.enums import DistanceType
from arekit.common.dataset.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.base import TextOpinion


class InputSampleBase(object):
    """
    Description of a single sample (context) of a model
    """

    def __init__(self, input_sample_id, values):
        assert(isinstance(input_sample_id, unicode))
        assert(isinstance(values, list))

        self.__input_sample_id = input_sample_id
        self.__values = OrderedDict(values)

    # region properties

    @property
    def ID(self):
        return self.__input_sample_id

    # endregion

    @staticmethod
    def check_ability_to_create_sample(window_size, text_opinion, text_opinion_helper):
        """
        Main text_opinion filtering rules
        """
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(text_opinion_helper, TextOpinionHelper))

        is_not_same_ends = False
        is_in_window = False
        is_same_sentence = False

        if text_opinion.SourceId != text_opinion.TargetId:
            is_not_same_ends = True

        if text_opinion_helper.calc_dist_between_text_opinion_ends(text_opinion, DistanceType.InTerms) < window_size:
            is_in_window = True

        if text_opinion_helper.calc_dist_between_text_opinion_ends(text_opinion, DistanceType.InSentences) == 0:
            is_same_sentence = True

        return is_not_same_ends and is_in_window and is_same_sentence

    def __iter__(self):
        for key, value in self.__values.iteritems():
            yield key, value
