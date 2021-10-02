import pandas as pd

from arekit.common.data import const
from arekit.common.utils import filter_whitespaces, split_by_whitespaces
from . import const as network_input_const

empty_list = []


def __process_indices_list(value):
    return [int(v) for v in str(value).split(network_input_const.ArgsSep)]


def __process_int_values_list(value):
    return __process_indices_list(value)


parse_value = {
    const.ID: lambda value: value,
    const.S_IND: lambda value: value,
    const.T_IND: lambda value: value,
    network_input_const.FrameVariantIndices: lambda value:
        __process_indices_list(value) if isinstance(value, str) else empty_list,
    network_input_const.FrameRoles: lambda value:
        __process_indices_list(value) if isinstance(value, str) else empty_list,
    network_input_const.SynonymObject: lambda value: __process_indices_list(value),
    network_input_const.SynonymSubject: lambda value: __process_indices_list(value),
    network_input_const.Entities: lambda value: __process_indices_list(value),
    network_input_const.PosTags: lambda value: __process_int_values_list(value),
    "text_a": lambda value: filter_whitespaces([term for term in split_by_whitespaces(value)])
}


class ParsedSampleRow(object):
    """
    Provides a parsed information for a sample row.
    TODO. Use this class as API
    """

    def __init__(self, row):
        assert(isinstance(row, pd.Series))

        self.__uint_label = None
        self.__params = {}

        for key, value in row.items():

            if key == const.LABEL:
                self.__uint_label = value
                continue

            if key not in parse_value:
                continue

            self.__params[key] = parse_value[key](value)

    @property
    def SampleID(self):
        return self.__params[const.ID]
    
    @property
    def Terms(self):
        return self.__params["text_a"]

    @property
    def SubjectIndex(self):
        return self.__params[const.S_IND]

    @property
    def ObjectIndex(self):
        return self.__params[const.T_IND]

    @property
    def UintLabel(self):
        return self.__uint_label

    @property
    def PartOfSpeechTags(self):
        return self.__params[network_input_const.PosTags]

    @property
    def TextFrameVariantIndices(self):
        return self.__params[network_input_const.FrameVariantIndices]

    @property
    def TextFrameVariantRoles(self):
        return self.__params[network_input_const.FrameRoles]

    @property
    def EntityInds(self):
        return self.__params[network_input_const.Entities]

    @property
    def SynonymObjectInds(self):
        return self.__params[network_input_const.SynonymObject]

    @property
    def SynonymSubjectInds(self):
        return self.__params[network_input_const.SynonymSubject]

    @classmethod
    def parse(cls, row):
        return cls(row=row)
