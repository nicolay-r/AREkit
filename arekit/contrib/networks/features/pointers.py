from . import utils


class PointersFeature(object):

    def __init__(self, vector_value):
        self.__vector_value = vector_value

    @classmethod
    def create_shifted_and_fit(cls, original_value, start_offset, end_offset, filler,
                               expected_size=None):

        modified_value = PointersFeature.__shift_text_pointers(
            inds=original_value,
            begin=start_offset,
            end=end_offset,
            pad_value=filler)

        if expected_size is not None:
            cls.__fit_inplace(value=modified_value,
                              size=expected_size,
                              filler=filler)

        return cls(vector_value=modified_value)

    @property
    def ValueVector(self):
        return self.__vector_value

    # region private methods

    @staticmethod
    def __shift_text_pointers(inds, begin, end, pad_value):
        return [PointersFeature.__shift_index(w_begin=begin, w_end=end,
                                                               index=index,
                                                               default=pad_value) for index in inds]

    @staticmethod
    def __fit_inplace(value, size, filler):
        if len(value) < size:
            utils.pad_right_inplace(lst=value,
                                    pad_size=size,
                                    filler=filler)
        else:
            del value[size:]

    @staticmethod
    def __shift_index(w_begin, w_end, index, default):
        shifted = index - w_begin
        in_window = utils.in_window(window_begin=w_begin,
                                    window_end=w_end,
                                    ind=index)
        return default if not in_window else shifted

    # endregion
