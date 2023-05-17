import json

from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.rusentiframes.effect import FrameEffect
from arekit.contrib.source.rusentiframes.io_utils import RuSentiFramesIOUtils
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.polarity import RuSentiFramesFramePolarity
from arekit.contrib.source.rusentiframes.role import FrameRole
from arekit.contrib.source.rusentiframes.state import FrameState


class RuSentiFramesCollection(object):

    __frames_key = "frames"
    __polarity_key = "polarity"
    __state_key = "state"
    __effect_key = "effect"
    __variants_key = "variants"

    def __init__(self, data, labels_fmt, effect_labels_fmt, lowercase_variants=True):
        """ data: dict
                Has the following structure of the frame contents:
                {
                    "frame_id": [ ... variants string list ... ]
                    ...
                }
            lowercase_variants: bool
                If 'True', forcely treat frame-variants as case-insensitive (lowercased)
                or avoiding lowercasing operation in case of 'False'.
        """
        assert(isinstance(data, dict))
        assert(isinstance(labels_fmt, StringLabelsFormatter))
        assert(isinstance(effect_labels_fmt, StringLabelsFormatter))
        self.__labels_fmt = labels_fmt
        self.__effect_labels_fmt = effect_labels_fmt
        self.__data = data

        if lowercase_variants:
            for frame_id, frame in self.__data.items():
                frame[self.__variants_key] = [variant.lower() for variant in frame[self.__variants_key]]

    # region classmethods

    @classmethod
    def read(cls, version, labels_fmt, effect_labels_fmt):
        assert(isinstance(version, RuSentiFramesVersions))
        assert(isinstance(labels_fmt, RuSentiFramesLabelsFormatter))
        assert(isinstance(effect_labels_fmt, RuSentiFramesEffectLabelsFormatter))

        return RuSentiFramesIOUtils.read_from_zip(
            inner_path=RuSentiFramesIOUtils.get_collection_filepath(),
            process_func=lambda input_file: cls.__from_json(
                input_file=input_file,
                labels_fmt=labels_fmt,
                effect_labels_fmt=effect_labels_fmt),
            version=version)

    @classmethod
    def __from_json(cls, input_file, labels_fmt, effect_labels_fmt):
        data = json.load(input_file)
        return cls(data=data,
                   labels_fmt=labels_fmt,
                   effect_labels_fmt=effect_labels_fmt)

    # endregion

    # region public 'try get' methods

    def try_get_frame_polarity(self, frame_id, role_src, role_dest):
        assert(isinstance(role_src, str))
        assert(isinstance(role_dest, str))

        if not self.__check_has_frame_polarity_key(frame_id):
            return None

        for args in self.__data[frame_id][self.__frames_key][self.__polarity_key]:
            if args[0] == role_src and args[1] == role_dest:
                return self.__frame_polarity_from_args(args)
        return None

    # endregion

    # region public 'get' methods

    def get_frame_roles(self, frame_id):
        assert(isinstance(frame_id, str))
        return [FrameRole(source=key, description=value)
                for key, value in self.__data[frame_id]["roles"].items()]

    def get_frame_polarities(self, frame_id):
        assert(isinstance(frame_id, str))

        if not self.__check_has_frame_polarity_key(frame_id):
            return []

        return [self.__frame_polarity_from_args(args)
                for args in self.__data[frame_id][self.__frames_key][self.__polarity_key]]

    def get_frame_states(self, frame_id):
        assert(isinstance(frame_id, str))

        if self.__state_key not in self.__data[frame_id][self.__frames_key]:
            return []

        return [FrameState(role=args[0], label=self.__labels_fmt.str_to_label(args[1]), prob=args[2])
                for args in self.__data[frame_id][self.__frames_key][self.__state_key]]

    def get_frame_titles(self, frame_id):
        assert(isinstance(frame_id, str))
        return self.__data[frame_id]["title"]

    def get_frame_variants(self, frame_id):
        return self.__data[frame_id][self.__variants_key]

    def get_frame_values(self, frame_id):
        assert(isinstance(frame_id, str))
        # TODO. Not implemented yet.
        pass

    def get_frame_effects(self, frame_id):
        assert(isinstance(frame_id, str))

        if self.__effect_key not in self.__data[frame_id][self.__frames_key]:
            return []

        return [FrameEffect(role=args[0], label=self.__effect_labels_fmt.str_to_label(args[1]), prob=args[2])
                for args in self.__data[frame_id][self.__frames_key][self.__effect_key]]

    # endregion

    # region public 'iter' methods

    def iter_frames_ids(self):
        for frame_id in self.__data.keys():
            yield frame_id

    def iter_frame_id_and_variants(self):
        for id, frame in self.__data.items():
            for variant in frame[self.__variants_key]:
                yield id, variant

    # endregion

    # region private methods

    def __check_has_frame_polarity_key(self, frame_id):
        return self.__polarity_key in self.__data[frame_id][self.__frames_key]

    def __frame_polarity_from_args(self, args):
        return RuSentiFramesFramePolarity(role_src=args[0],
                                          role_dest=args[1],
                                          label=self.__labels_fmt.str_to_label(args[2]),
                                          prob=args[3])

    # endregion
