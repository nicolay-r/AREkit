# -*- coding: utf-8 -*-
from core.processing.lemmatization.base import Stemmer


class FramesCollection:

    def __init__(self, variants, frames_list):
        """
        frames: dict
            dictionary of frames: <str, Frame>
        """
        assert(isinstance(variants, dict))
        self.__variants = variants
        self.__frames_list = frames_list

    @classmethod
    def from_file(cls, filepath, stemmer=None):
        """
        Reads SentiRuFrames collection -- list of variants with related frame groups.
        """
        assert(isinstance(stemmer, Stemmer) or stemmer is None)

        frames = {}
        frames_dict = {}
        frames_list = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.decode('utf-8')
                char = u'–' if u'–' in line else u'-'

                assert(len(filter(lambda c: c == char, line)) == 1)

                separator = line.index(char)

                template = line[0:separator].strip().lower()
                if stemmer is not None:
                    template = stemmer.lemmatize_to_str(template)

                if u',' in template:
                    print template
                    raise Exception(template)

                groups = line[separator + 1:].strip()
                groups = groups[5:] if u'фрейм' in groups else groups
                groups = [g.strip().lower() for g in groups.split(u',')]

                indices = [cls.__add_frame(frames_dict, frames_list, g) for g in groups]

                frames[template] = Variant(template, indices)

        return cls(frames, frames_list)

    @staticmethod
    def __add_frame(frames_dict, frames_list, group):
        assert(isinstance(group, unicode))
        if group not in frames_dict:
            frames_dict[group] = len(frames_list)
            frames_list.append(group)
        return frames_dict[group]

    def get_frame_by_index(self, index):
        return self.__frames_list[index]

    def get_variant_by_template(self, template):
        return self.__variants[template]

    def has_variant(self, template):
        return template in self.__variants

    def iter_variants(self):
        for template, variant in self.__variants.iteritems():
            yield template, variant


class Variant:

    def __init__(self, template, frame_indices):
        assert(isinstance(template, unicode))
        self.terms = template.lower().split()
        self.frame_indices = frame_indices


class VariantInText:

    def __init__(self, variant, start_index):
        assert(isinstance(variant, Variant))
        assert(isinstance(start_index, int))
        self.__variant = variant
        self.start_index = start_index

    def get_bounds(self):
        return self.start_index, self.start_index + len(self)

    def iter_terms(self):
        for term in self.__variant.terms:
            yield term

    def __len__(self):
        return len(self.__variant.terms)

