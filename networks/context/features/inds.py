import collections
import utils


class IndicesFeature:

    def __init__(self, value_vector, start_index, end_index):
        self.__value_vector = value_vector
        self.__start_index = start_index
        self.__end_index = end_index

    @classmethod
    def from_vector_to_be_fitted(cls, value_vector, e1_in, e2_in, expected_size, filler):
        assert(isinstance(value_vector, collections.Iterable))
        assert(isinstance(expected_size, int))
        assert(0 <= e1_in < expected_size)
        assert(0 <= e2_in < expected_size)

        start_index = 0
        value_modified = list(value_vector)

        if len(value_modified) < expected_size:
            utils.pad_right_inplace(value_modified,
                                    pad_size=expected_size,
                                    filler=filler)
            end_index = expected_size
        else:
            start_index, end_index = cls.__calculate_bounds(
                window_size=expected_size,
                e1=e1_in,
                e2=e2_in)

            cls.__crop_inplace(lst=value_modified,
                               begin=start_index,
                               end=end_index)

        return cls(value_vector=value_modified,
                   start_index=start_index,
                   end_index=end_index)

    # region properties

    @property
    def StartIndex(self):
        return self.__start_index

    @property
    def EndIndex(self):
        return self.__end_index

    @property
    def ValueVector(self):
        return self.__value_vector

    # endregion

    # region private methods

    @staticmethod
    def __calculate_bounds(window_size, e1, e2):
        assert(isinstance(window_size, int) and window_size > 0)
        w_begin = 0
        w_end = window_size
        while not (utils.in_window(window_begin=w_begin, window_end=w_end, ind=e1) and
                   utils.in_window(window_begin=w_begin, window_end=w_end, ind=e2)):
            w_begin += 1
            w_end += 1

        return w_begin, w_end

    @staticmethod
    def __crop_inplace(lst, begin, end):
        if end < len(lst):
            del lst[end:]
        del lst[:begin]

    # endregion
