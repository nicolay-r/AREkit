class FrameVariant(object):

    def __init__(self, terms, frame_id):
        assert(isinstance(terms, list))
        assert(isinstance(frame_id, str))
        self.__terms = terms
        self.__frame_id = frame_id

    @property
    def FrameID(self):
        return self.__frame_id

    def get_value(self):
        return " ".join(self.__terms)

    def iter_terms(self):
        for term in self.__terms:
            yield term

    def __len__(self):
        return len(self.__terms)