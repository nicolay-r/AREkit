from arekit.common.entities.types import OpinionEntityType


class StringEntitiesFormatter(object):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, OpinionEntityType))
        raise NotImplementedError()

    @staticmethod
    def iter_supported_types():
        for entity_type in OpinionEntityType:
            yield entity_type