class Entity(object):

    def __init__(self, value, e_type, group_index=None):
        assert(isinstance(value, str) and len(value) > 0)
        assert(isinstance(e_type, str) or e_type is None)
        assert(isinstance(group_index, int) or group_index is None)
        self.__value = value.lower()
        self.__type = e_type
        self.__group_index = group_index
        self.__display_value = None

    @property
    def GroupIndex(self):
        return self.__group_index

    @property
    def Value(self):
        return self.__value

    @property
    def DisplayValue(self):
        """ Now, we consider the default value in case
            of the undefined caption, and display_value otherwise.
        """
        return self.__value if self.__display_value is None else self.__display_value

    @property
    def Type(self):
        return self.__type

    def set_display_value(self, caption):
        """ Caption allows to customize the original value.
            Required for optional value modification.
        """
        assert(isinstance(caption, str))
        self.__display_value = caption

    def set_group_index(self, value):
        assert(isinstance(value, int) and value >= -1)
        assert(self.__group_index is None)
        self.__group_index = value
