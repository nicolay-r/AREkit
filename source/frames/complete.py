import json

from core.evaluation.labels import Label


class FramesCollection:

    __frames_key = u"frames"
    __polarity_key = u"polarity"
    __states_key = u"states"

    def __init__(self, data):
        assert(isinstance(data, dict))
        self.__data = data

    @classmethod
    def from_json(cls, filepath):
        assert(isinstance(filepath, unicode))
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(data)

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

    def try_get_frame_polarity(self, frame_id, role_src, role_dest):
        assert(isinstance(role_src, unicode))
        assert(isinstance(role_dest, unicode))

        if not self.__check_has_frame_polarity_key(frame_id):
            return None

        for args in self.__data[frame_id][self.__frames_key][self.__polarity_key]:
            if args[0] == role_src and args[1] == role_dest:
                return self.__frame_polarity_from_args(args)
        return None

    def get_frame_states(self, frame_id):
        assert(isinstance(frame_id, unicode))

        if self.__states_key not in self.__data[frame_id][self.__frames_key]:
            return []

        return [FrameState(role=args[0], label=args[1], prob=args[2])
                for args in self.__data[frame_id][self.__frames_key][self.__states_key]]

    def get_frame_titles(self, frame_id):
        assert(isinstance(frame_id, unicode))
        return self.__data[frame_id][u"title"]

    def get_frame_values(self, frame_id):
        assert(isinstance(frame_id, unicode))
        pass

    def get_frame_variants(self, frame_id):
        return self.__data[frame_id][u"variants"]

    def get_frame_effects(self, frame_id):
        assert(isinstance(frame_id, unicode))
        pass

    def iter_frames_ids(self):
        for frame_id in self.__data.iterkeys():
            yield frame_id

    def __check_has_frame_polarity_key(self, frame_id):
        return self.__polarity_key in self.__data[frame_id][self.__frames_key]

    @staticmethod
    def __frame_polarity_from_args(args):
        return FramePolarity(src=args[0], dest=args[1], label=Label.from_str(args[2]), prob=args[3])

    def iter_frame_id_and_variants(self):
        for id, frame in self.__data.iteritems():
            for variant in frame["variants"]:
                yield id, variant


class FramePolarity(object):

    def __init__(self, src, dest, label, prob):
        assert(isinstance(src, unicode))
        assert(isinstance(dest, unicode))
        assert(isinstance(label, Label))
        assert(isinstance(prob, float))
        self.__src = src
        self.__dest = dest
        self.__label = label
        self.__prob = prob

    @property
    def Source(self):
        return self.__src

    @property
    def Destination(self):
        return self.__dest

    @property
    def Label(self):
        return self.__label

    @property
    def Prob(self):
        return self.__prob


class FrameState(object):

    def __init__(self, role, label, prob):
        assert(isinstance(role, unicode))
        assert(isinstance(label, Label))
        assert(isinstance(prob, float))
        self.__role = role
        self.__label = label
        self.__prob = prob

    @property
    def Role(self):
        return self.__role

    @property
    def Label(self):
        return self.__label

    @property
    def Prob(self):
        return self.__prob


class FrameRole(object):

    def __init__(self, source, description):
        assert(isinstance(source, unicode))
        assert(isinstance(description, unicode))
        self.__source = source
        self.__description = description

    @property
    def Source(self):
        return self.__source

    @property
    def Description(self):
        return self.__description


class FrameEffect(object):
    pass


class FrameValue(object):
    pass
