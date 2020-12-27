from arekit.common.entities.base import Entity
from arekit.common.synonyms import SynonymsCollection


def entity_to_group_func(entity, synonyms):
    assert(isinstance(entity, Entity) or entity is None)
    assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)

    if entity is None:
        return None

    if synonyms is None:
        return None

    value = entity.Value

    if not synonyms.contains_synonym_value(value):
        return None

    return synonyms.get_synonym_group_index(value)