from arekit.common.entities.types import OpinionEntityType


class StringEntitiesFormatter(object):

    def to_string(self, original_value, entity_type):
        assert(isinstance(entity_type, OpinionEntityType))
        raise NotImplementedError()
