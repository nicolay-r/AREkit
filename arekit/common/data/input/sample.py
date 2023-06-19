from collections import OrderedDict

from arekit.common.docs.parsed.providers.entity_service import EntityServiceProvider, DistanceType
from arekit.common.text_opinions.base import TextOpinion


class InputSampleBase(object):
    """
    Description of a single sample (context) of a model
    """

    def __init__(self, shift_index_dbg, input_sample_id, values):
        assert(isinstance(shift_index_dbg, int))
        assert(isinstance(input_sample_id, str))
        assert(isinstance(values, list))
        self._shift_index_dbg = shift_index_dbg
        self.__input_sample_id = input_sample_id
        self.__values = OrderedDict(values)

    # region properties

    @property
    def ID(self):
        return self.__input_sample_id

    # endregion

    @staticmethod
    def check_ability_to_create_sample(entity_service, window_size, text_opinion):
        """
        Main text_opinion filtering rules
        """
        assert(isinstance(entity_service, EntityServiceProvider))
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(window_size, int) and window_size > 0)

        is_not_same_ends = False
        is_in_window = False
        is_same_sentence = False

        if text_opinion.SourceId != text_opinion.TargetId:
            is_not_same_ends = True

        dist_between_entities = entity_service.calc_dist_between_text_opinion_ends(
            text_opinion=text_opinion,
            distance_type=DistanceType.InTerms)

        if InputSampleBase._check_ends_could_be_fitted_in_window(dist_between_entities, window_size):
            is_in_window = True

        dist_in_sents = entity_service.calc_dist_between_text_opinion_ends(
            text_opinion=text_opinion,
            distance_type=DistanceType.InSentences)

        if dist_in_sents == 0:
            is_same_sentence = True

        return is_not_same_ends and is_in_window and is_same_sentence

    @staticmethod
    def _check_ends_could_be_fitted_in_window(actual_dist, window):
        return actual_dist < window

    def __iter__(self):
        for key, value in self.__values.items():
            yield key, value
