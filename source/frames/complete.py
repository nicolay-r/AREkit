import json

from core.evaluation.labels import Label


class FramesCollection:

    __frames_key = u"frames"
    __polarity_key = u"polarity"

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
        return self.__data[frame_id]["roles"]

    def get_frame_polarities(self, frame_id):
        assert(isinstance(frame_id, unicode))

        if self.__polarity_key not in self.__data[frame_id][self.__frames_key]:
            return []

        return [FramePolarity(src=args[0], dest=args[1], label=Label.from_str(args[2]), prob=args[3])
                for args in self.__data[frame_id][self.__frames_key][self.__polarity_key]]

    def get_frame_states(self, frame_id):
        assert(isinstance(frame_id, unicode))
        return [FrameState(role=args[0], label=args[1], prob=args[2])
                for args in self.__data[frame_id][self.__frames_key]["states"]]

    def get_frame_values(self):
        pass

    def get_frame_effects(self):
        pass

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


class FrameValues(object):
    pass


class FrameEffect(object):
    pass
