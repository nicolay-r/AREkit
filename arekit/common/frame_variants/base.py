class FrameVariant(object):

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