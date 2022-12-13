import pandas as pd

from arekit.common.data import const
from arekit.common.utils import filter_whitespaces, split_by_whitespaces
from . import const as network_input_const

empty_list = []


def __process_values_list(value):
    return value.split(network_input_const.ArgsSep)


def __process_indices_list(value):
    return [int(v) for v in str(value).split(network_input_const.ArgsSep)]


def __process_int_values_list(value):
    return __process_indices_list(value)


parse_value = {
    const.ID: lambda value: value,
    const.DOC_ID: lambda value: int(value),
    const.S_IND: lambda value: int(value),
    const.T_IND: lambda value: int(value),
    const.SENT_IND: lambda value: int(value),
    const.ENTITY_VALUES: lambda value: __process_values_list(value),
    const.ENTITY_TYPES: lambda value: __process_values_list(value),
    const.ENTITIES: lambda value: __process_indices_list(value),
    network_input_const.FrameVariantIndices: lambda value:
        __process_indices_list(value) if isinstance(value, str) else empty_list,
    network_input_const.FrameConnotations: lambda value:
        __process_indices_list(value) if isinstance(value, str) else empty_list,
    network_input_const.SynonymObject: lambda value: __process_indices_list(value),
    network_input_const.SynonymSubject: lambda value: __process_indices_list(value),
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
                self.__uint_label = int(value)
                # TODO: To be adopted in future instead of __uint_label
                self.__params[key] = value
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
    def TextFrameConnotations(self):
        return self.__params[network_input_const.FrameConnotations]

    @property
    def EntityInds(self):
        return self.__params[const.ENTITIES]

    @property
    def SynonymObjectInds(self):
        return self.__params[network_input_const.SynonymObject]

    @property
    def SynonymSubjectInds(self):
        return self.__params[network_input_const.SynonymSubject]

    def __getitem__(self, item):
        assert (isinstance(item, str) or item is None)
        if item not in self.__params:
            return None
        return self.__params[item] if item is not None else None

    @classmethod
    def parse(cls, row):
        return cls(row=row)
