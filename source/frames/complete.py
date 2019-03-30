import json


class FramesCollection:

    def __init__(self, data):
        self.__data = data

    @classmethod
    def from_json(cls, filepath):
        assert(isinstance(filepath, unicode))
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(data)

    def get_roles(self):
        pass



