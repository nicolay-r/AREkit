import utils


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
            return cls.__fit_inplace(value=modified_value,
                                     size=expected_size,
                                     filler=filler)

        return cls(vector_value=modified_value)

    @property
    def ValueVector(self):
        return self.__vector_value

    # region private methods

    @staticmethod
    def __shift_text_pointers(inds, begin, end, pad_value):
        return map(lambda frame_index: PointersFeature.__shift_index(w_begin=begin, w_end=end,
                                                                     frame_index=frame_index,
                                                                     placeholder=pad_value),
                   inds)

    @staticmethod
    def __fit_inplace(value, size, filler):
        if len(value) < size:
            utils.pad_right_inplace(lst=value,
                                    pad_size=size,
                                    filler=filler)
        else:
            del value[size:]


    @staticmethod
    def __shift_index(w_begin, w_end, frame_index, placeholder):
        shifted = frame_index - w_begin
        in_window = utils.in_window(window_begin=w_begin,
                                    window_end=w_end,
                                    ind=frame_index)
        return placeholder if not in_window else shifted

    # endregion
