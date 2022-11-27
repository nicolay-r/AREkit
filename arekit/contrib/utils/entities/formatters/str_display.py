from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter


class StringEntitiesDisplayValueFormatter(StringEntitiesFormatter):
    """ Provides the contents of the DisplayValue property.
    """

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        return original_value.DisplayValue
