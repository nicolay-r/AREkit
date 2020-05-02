from collections import OrderedDict

from arekit.common.text_opinions.helper import TextOpinionHelper


class InputSampleBase(object):
    """
    Description of a single sample (context) of a model
    """

    def __init__(self, text_opinion_id, values):
        assert(isinstance(text_opinion_id, int))
        assert(isinstance(values, list))

        self.__text_opinion_id = text_opinion_id
        self.__values = OrderedDict(values)

    # region properties

    @property
    def TextOpinionID(self):
        return self.__text_opinion_id

    # endregion

    @staticmethod
    def check_ability_to_create_sample(window_size, text_opinion):
        return abs(TextOpinionHelper.calculate_distance_between_entities_in_terms(text_opinion)) < window_size

    def __iter__(self):
        for key, value in self.__values.iteritems():
            yield key, value
