class ParsedSampleRow(object):
    """ Provides a parsed information for a sample row.
    """

    def __init__(self, row, columns_fmts, no_value_func):
        """ row: dict
                dict of the pairs ("field_name", value)
            columns_fmt: list
                list of the formatters, where every formatter represent a dictionary.
            no_value_func: func
                the default value the conveys the absence of the parameter-value.
        """
        assert(isinstance(row, dict))
        assert(isinstance(columns_fmts, list))
        assert(callable(no_value_func))

        self.__uint_label = None
        self.__params = {}
        self.__no_value = no_value_func

        for key, value in row.items():

            for columns_fmt in columns_fmts:
                assert(isinstance(columns_fmt, dict))

                if key not in columns_fmt:
                    continue

                self.__params[key] = columns_fmt[key](value)
                break

    def __value_or_none(self, key):
        return self.__params[key] if key in self.__params else self.__no_value()

    def __getitem__(self, item):
        assert (isinstance(item, str) or item is None)
        if item not in self.__params:
            return self.__no_value()
        return self.__params[item] if item is not None else self.__no_value()

    @classmethod
    def parse(cls, row, columns_fmts, no_value_func):
        return cls(row=row, columns_fmts=columns_fmts, no_value_func=no_value_func)
