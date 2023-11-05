from collections.abc import Iterable
from arekit.common.frames.variants.base import FrameVariant


class FrameVariantsCollection(object):

    def __init__(self):
        self.__variants = {}
        self.__frames_list = []

    # region private methods

    @staticmethod
    def __register_frame(frames_dict, frames_list, id):
        assert(isinstance(id, str))
        if id not in frames_dict:
            frames_dict[id] = len(frames_list)
            frames_list.append(id)
        return frames_dict[id]

    # endregion

    # region public methods

    def fill_from_iterable(self, variants_with_id, overwrite_existed_variant, raise_error_on_existed_variant):
        assert(isinstance(variants_with_id, Iterable))
        assert(isinstance(overwrite_existed_variant, bool))
        assert(isinstance(raise_error_on_existed_variant, bool))
        assert(len(self.__variants) == 0)
        assert(len(self.__frames_list) == 0)

        frames_dict = {}
        for frame_id, variant in variants_with_id:
            self.__register_frame(frames_dict, self.__frames_list, frame_id)

            if variant in self.__variants:
                if raise_error_on_existed_variant:
                    raise Exception("Variant '{variant}' already registered".format(variant=variant))
                if not overwrite_existed_variant:
                    continue

            self.__variants[variant] = FrameVariant(terms=variant.split(), frame_id=frame_id)

    def get_frame_by_index(self, index):
        return self.__frames_list[index]

    def get_variant_by_value(self, value):
        return self.__variants[value] if value in self.__variants else None

    def has_variant(self, value):
        return value in self.__variants

    def iter_variants(self):
        for value, variant in self.__variants.items():
            yield value, variant

    # endregion

    def __len__(self):
        return len(self.__variants)
