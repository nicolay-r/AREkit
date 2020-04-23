from collections import OrderedDict


class InputSampleBase(object):
    """
    Description of a single sample (context) of a model
    """

    def __init__(self, text_opinion_id, values):
        assert(isinstance(text_opinion_id, int))
        assert(isinstance(values, list))

        self.__text_opinion_id = text_opinion_id
        self.__values = OrderedDict(values)

    # region properties

    @property
    def TextOpinionID(self):
        return self.__text_opinion_id

    # endregion

    def __iter__(self):
        for key, value in self.__values.iteritems():
            yield key, value
