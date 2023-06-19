from arekit.common.data.input.providers.label.multiple import MultipleLabelProvider
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider


class CroppedSampleRowProvider(BaseSampleRowProvider):
    """ Sample provided which has `crop_window` that allows to slice
        the potentially large samples and guarantee the presence of
        attitude inside.
    """

    def __init__(self, crop_window_size, label_scaler, text_provider):
        assert(isinstance(crop_window_size, int) and crop_window_size > 0)
        super(CroppedSampleRowProvider, self).__init__(label_provider=MultipleLabelProvider(label_scaler),
                                                       text_provider=text_provider)
        self.__crop_window_size = crop_window_size

    @staticmethod
    def __calc_window_bounds(window_size, s_ind, t_ind, input_length):
        """ returns: [_from, _to)
        """
        assert(isinstance(s_ind, int))
        assert(isinstance(t_ind, int))
        assert(isinstance(input_length, int))
        assert(input_length >= s_ind and input_length >= t_ind)

        def __in():
            return _from <= s_ind < _to and _from <= t_ind < _to

        _from = 0
        _to = window_size
        while not __in():
            _from += 1
            _to += 1

        return _from, _to

    def _provide_sentence_terms(self, parsed_doc, sentence_ind, s_ind, t_ind):
        terms_iter, src_ind, tgt_ind = super(CroppedSampleRowProvider, self)._provide_sentence_terms(
            parsed_doc=parsed_doc, sentence_ind=sentence_ind, s_ind=s_ind, t_ind=t_ind)
        terms = list(terms_iter)
        _from, _to = self.__calc_window_bounds(window_size=self.__crop_window_size,
                                               s_ind=s_ind, t_ind=t_ind, input_length=len(terms))
        return terms[_from:_to], src_ind - _from, tgt_ind - _from
