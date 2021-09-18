from enum import Enum


class ModelInputType(Enum):
    SingleInstance = 'ctx'
    MultiInstanceMaxPooling = 'mi-mp'
    MultiInstanceWithSelfAttention = 'mi-self-att'


class ModelInputTypeService(object):

    __names = dict([(item.value, item) for item in ModelInputType])

    @staticmethod
    def __iter_supported_names():
        return iter(list(ModelInputTypeService.__names.keys()))

    @staticmethod
    def get_type_by_name(name):
        return ModelInputTypeService.__names[name]

    @staticmethod
    def find_name_by_type(input_type):
        assert(isinstance(input_type, ModelInputType))

        for name in ModelInputTypeService.__iter_supported_names():
            related_type = ModelInputTypeService.__names[name]
            if related_type == input_type:
                return name

    @staticmethod
    def iter_supported_names():
        return ModelInputTypeService.__iter_supported_names()
