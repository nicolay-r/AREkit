# -*- coding: utf-8 -*-
import collections

from core.common.bound import Bound
from core.processing.lemmatization.base import Stemmer


class FrameVariantsCollection:

    def __init__(self, variants, frames_list, stemmer):
        """
        frames_list: list
            list of "frame_id" (typeof unicode) items.
        """
        assert(isinstance(variants, dict))
        assert(isinstance(frames_list, list))
        assert(isinstance(stemmer, Stemmer))
        self.__variants = variants
        self.__lemma_variants = self.__create_lemmatized_variants(stemmer)
        self.__frames_list = frames_list

    def __create_lemmatized_variants(self, stemmer):
        assert(isinstance(stemmer, Stemmer))

        lemma_variants = {}
        for variant, frame_variant in self.__variants.iteritems():
            key = stemmer.lemmatize_to_str(variant)
            if key in lemma_variants:
                continue
            lemma_variants[key] = frame_variant

        return lemma_variants

    @classmethod
    def from_iterable(cls, variants_with_id, stemmer):
        assert(isinstance(variants_with_id, collections.Iterable))
        assert(isinstance(stemmer, Stemmer))

        variants = {}
        frames_dict = {}
        frames_list = []
        for frame_id, variant in variants_with_id:
            FrameVariantsCollection.__register_frame(frames_dict, frames_list, frame_id)
            variants[variant] = FrameVariant(variant, frame_id)

        return cls(variants=variants, frames_list=frames_list, stemmer=stemmer)

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
        if template in self.__variants:
            return self.__variants[template]
        return self.__lemma_variants[template]

    def has_variant(self, template):
        if template in self.__variants:
            return True
        return template in self.__lemma_variants

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

    def __init__(self, variant, start_index, is_inverted):
        assert(isinstance(variant, FrameVariant))
        assert(isinstance(start_index, int))
        assert(isinstance(is_inverted, bool))
        self.__variant = variant
        self.__start_index = start_index
        self.__is_inverted = is_inverted

    @property
    def Variant(self):
        return self.__variant

    @property
    def Position(self):
        return self.__start_index

    @property
    def IsInverted(self):
        return self.__is_inverted

    def get_bound(self):
        return Bound(pos=self.__start_index, length=len(self))

    def iter_terms(self):
        for term in self.__variant.iter_terms():
            yield term

    def __len__(self):
        return len(self.__variant)

