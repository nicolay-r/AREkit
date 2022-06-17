from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType


class StringEntitiesSimpleFormatter(StringEntitiesFormatter):
    """
    Utilized for picking a related word in word embedding.
    """

    def to_string(self, original_value, entity_type):
        """
        Returns: str (unicode)
            Value that assumes to be utilized in Word2Vec model embedding search.
        """
        assert(isinstance(entity_type, OpinionEntityType))

        if entity_type == OpinionEntityType.Other:
            return "e"
        elif entity_type == OpinionEntityType.Object or entity_type == OpinionEntityType.SynonymObject:
            return "object"
        elif entity_type == OpinionEntityType.Subject or entity_type == OpinionEntityType.SynonymSubject:
            return "subject"

        return None
