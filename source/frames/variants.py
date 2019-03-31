# -*- coding: utf-8 -*-
import collections

from core.common.bound import Bound


class FrameVariantsCollection:

    def __init__(self, variants, frames_list):
        """
        frames_list: list
            list of "frame_id" (typeof unicode) items.
        """
        assert(isinstance(variants, dict))
        assert(isinstance(frames_list, list))
        self.__variants = variants
        self.__frames_list = frames_list

    @classmethod
    def from_iterable(cls, variants_with_id):
        assert(isinstance(variants_with_id, collections.Iterable))

        variants = {}
        frames_dict = {}
        frames_list = []
        for frame_id, variant in variants_with_id:
            FrameVariantsCollection.__register_frame(frames_dict, frames_list, frame_id)
            variants[variant] = FrameVariant(variant, frame_id)

        return cls(variants, frames_list)

    @staticmethod
    def __register_frame(frames_dict, frames_list, id):
        assert(isinstance(id, unicode))
        if id not in frames_dict:
            frames_dict[id] = len(frames_list)
            frames_list.append(id)
        return frames_dict[id]

    def get_frame_by_index(self, index):
        return self.__frames_list[index]

    def get_variant_by_template(self, template):
        return self.__variants[template]

    def has_variant(self, template):
        return template in self.__variants

    def iter_variants(self):
        for template, variant in self.__variants.iteritems():
            yield template, variant


class FrameVariant:

    def __init__(self, text, frame_id):
        assert(isinstance(text, unicode))
        assert(isinstance(frame_id, unicode))
        self.__terms = text.lower().split()
        self.__frame_id = frame_id

    @property
    def FrameID(self):
        return self.__frame_id

    def get_value(self):
        return u" ".join(self.__terms)

    def iter_terms(self):
        for term in self.__terms:
            yield term

    def __len__(self):
        return len(self.__terms)


class FrameVariantInText:

    def __init__(self, variant, start_index):
        assert(isinstance(variant, FrameVariant))
        assert(isinstance(start_index, int))
        self.__variant = variant
        self.start_index = start_index

    @property
    def Variant(self):
        return self.__variant

    # TODO: Deprecated. Use get_bound instead.
    def get_bounds(self):
        return self.start_index, self.start_index + len(self)

    def get_bound(self):
        return Bound(pos=self.start_index, length=len(self))

    def iter_terms(self):
        for term in self.__variant.iter_terms():
            yield term

    def __len__(self):
        return len(self.__variant)

