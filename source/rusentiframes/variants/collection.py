# -*- coding: utf-8 -*-
import collections
from core.processing.lemmatization.base import Stemmer
from core.source.rusentiframes.variants.variant import FrameVariant


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

