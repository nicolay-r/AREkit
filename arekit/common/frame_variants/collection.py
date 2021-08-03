import collections
from arekit.common.frame_variants.base import FrameVariant


class FrameVariantsCollection(object):

    def __init__(self):
        self.__variants = {}
        self.__frames_list = []

    @property
    def Variants(self):
        return self.__variants

    # region private methods

    @staticmethod
    def __register_frame(frames_dict, frames_list, id):
        assert(isinstance(id, unicode))
        if id not in frames_dict:
            frames_dict[id] = len(frames_list)
            frames_list.append(id)
        return frames_dict[id]

    # endregion

    # region public methods

    def fill_from_iterable(self, variants_with_id):
        assert(isinstance(variants_with_id, collections.Iterable))
        assert(len(self.__variants) == 0)
        assert(len(self.__frames_list) == 0)

        __check_uniqueness = False

        frames_dict = {}
        for frame_id, variant in variants_with_id:
            self.__register_frame(frames_dict, self.__frames_list, frame_id)

            if variant in self.__variants and __check_uniqueness:
                raise Exception("Variant already registered")

            self.__variants[variant] = FrameVariant(variant, frame_id)

    def get_frame_by_index(self, index):
        return self.__frames_list[index]

    def get_variant_by_value(self, value):
        return self.__variants[value] if value in self.__variants else None

    def has_variant(self, value):
        return value in self.__variants

    def iter_variants(self):
        for value, variant in self.__variants.iteritems():
            yield value, variant

    # endregion
