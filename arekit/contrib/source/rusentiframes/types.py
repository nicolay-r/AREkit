from enum import Enum


class RuSentiFramesVersions(Enum):

    # Papers for description:
    # Distant Supervision for Sentiment Attitude Extraction (RANLP-2019)
    # Nicolay Rusnachenko, Natalia Loukachevitch, Elena Tutubalina
    # https://www.aclweb.org/anthology/R19-1118/
    # https://github.com/nicolay-r/RuSentiFrames/tree/v1.0
    V10 = "v1_0"

    # Papers for description:
    # Sentiment Frames for Attitude Extraction in Russian (DIALOG-2020)
    # Natalia Loukachevitch, Nicolay Rusnachenko
    # https://github.com/nicolay-r/RuSentiFrames/tree/v2.0
    V20 = "v2_0"


class RuSentiFramesVersionsService:

    @staticmethod
    def __iter_supported_types():
        return iter(RuSentiFramesVersions)

    @staticmethod
    def get_name_by_type(version_type):
        assert(isinstance(version_type, RuSentiFramesVersions))
        return version_type.value

    @staticmethod
    def get_type_by_name(name):
        for version_type in RuSentiFramesVersionsService.__iter_supported_types():
            if version_type.value == name:
                return version_type

        raise Exception("RuSentiFrames version by name `{}` was hot found!".format(name))

    @staticmethod
    def iter_supported_names():
        for version_type in RuSentiFramesVersionsService.__iter_supported_types():
            yield version_type.value