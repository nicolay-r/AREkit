import json

from core.evaluation.labels import Label
from core.common.frames.collection import FramesCollection
from core.source.rusentiframes.io_utils import RuSentiFramesIOUtils
from core.source.rusentiframes.polarity import RuSentiFramesFramePolarity
from core.source.rusentiframes.role import FrameRole
from core.source.rusentiframes.state import FrameState


class RuSentiFramesCollection(FramesCollection):

    __frames_key = u"frames"
    __polarity_key = u"polarity"
    __state_key = u"state"

    def __init__(self, data):
        assert(isinstance(data, dict))
        self.__data = data

    # region classmethods

    @classmethod
    def read_collection(cls):
        return RuSentiFramesIOUtils.read_from_zip(
            inner_path=RuSentiFramesIOUtils.get_collection_filepath(),
            process_func=lambda input_file: cls.__from_json(input_file))

    @classmethod
    def __from_json(cls, input_file):
        data = json.load(input_file)
        return cls(data)

    # endregion

    # region public 'try get' methods

    def try_get_frame_sentiment_polarity(self, frame_id):
        return self.try_get_frame_polarity(frame_id=frame_id,
                                           role_src=u'a0',
                                           role_dest=u'a1')

    def try_get_frame_polarity(self, frame_id, role_src, role_dest):
        assert(isinstance(role_src, unicode))
        assert(isinstance(role_dest, unicode))

        if not self.__check_has_frame_polarity_key(frame_id):
            return None

        for args in self.__data[frame_id][self.__frames_key][self.__polarity_key]:
            if args[0] == role_src and args[1] == role_dest:
                return self.__frame_polarity_from_args(args)
        return None

    # endregion

    # region public 'get' methods

    def get_frame_roles(self, frame_id):
        assert(isinstance(frame_id, unicode))
        return [FrameRole(source=key, description=value)
                for key, value in self.__data[frame_id][u"roles"].iteritems()]

    def get_frame_polarities(self, frame_id):
        assert(isinstance(frame_id, unicode))

        if not self.__check_has_frame_polarity_key(frame_id):
            return []

        return [self.__frame_polarity_from_args(args)
                for args in self.__data[frame_id][self.__frames_key][self.__polarity_key]]

    def get_frame_states(self, frame_id):
        assert(isinstance(frame_id, unicode))

        if self.__state_key not in self.__data[frame_id][self.__frames_key]:
            return []

        return [FrameState(role=args[0], label=Label.from_str(args[1]), prob=args[2])
                for args in self.__data[frame_id][self.__frames_key][self.__state_key]]

    def get_frame_titles(self, frame_id):
        assert(isinstance(frame_id, unicode))
        return self.__data[frame_id][u"title"]

    def get_frame_variants(self, frame_id):
        return self.__data[frame_id][u"variants"]

    def get_frame_values(self, frame_id):
        assert(isinstance(frame_id, unicode))
        # TODO. Not implemented yet.
        pass

    def get_frame_effects(self, frame_id):
        assert(isinstance(frame_id, unicode))
        # TODO. Not implemented yet.
        pass

    # endregion

    # region public 'iter' methods

    def iter_frames_ids(self):
        for frame_id in self.__data.iterkeys():
            yield frame_id

    def iter_frame_id_and_variants(self):
        for id, frame in self.__data.iteritems():
            for variant in frame["variants"]:
                yield id, variant

    # endregion

    # region private methods

    def __check_has_frame_polarity_key(self, frame_id):
        return self.__polarity_key in self.__data[frame_id][self.__frames_key]

    @staticmethod
    def __frame_polarity_from_args(args):
        return RuSentiFramesFramePolarity(role_src=args[0],
                                          role_dest=args[1],
                                          label=Label.from_str(args[2]),
                                          prob=args[3])

    # endregion
