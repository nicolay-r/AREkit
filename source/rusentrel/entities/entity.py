# TODO. To /common/collection.py
class Entity(object):
    """ Entity description.
    """

    def __init__(self, ID, str_type, begin, end, value):
        assert(type(ID) == unicode)
        assert(type(str_type) == unicode)
        assert(type(begin) == int)
        assert(type(end) == int)
        assert(type(value) == unicode and len(value) > 0)
        self.ID = ID
        self.str_type = str_type
        self.begin = begin
        self.end = end
        self.value = value.lower()

    def get_int_ID(self):
        return int(self.ID[1:len(self.ID)])