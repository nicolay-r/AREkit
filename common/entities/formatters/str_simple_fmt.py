from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import EntityType


class StringSimpleFormatter(StringEntitiesFormatter):
    """
    Utilized for picking a related word in word embedding.
    """

    def to_string(self, original_value, entity_type):
        """
        Returns: str (unicode)
            Value that assumes to be utilized in Word2Vec model embedding search.
        """
        assert(isinstance(entity_type, EntityType))

        if entity_type == EntityType.Other:
            return u"e"
        elif entity_type == EntityType.Object or entity_type == EntityType.SynonymObject:
            return u"object"
        elif entity_type == EntityType.Subject or entity_type == EntityType.SynonymSubject:
            return u"subject"

        return None
