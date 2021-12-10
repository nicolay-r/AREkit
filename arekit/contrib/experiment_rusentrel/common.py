from arekit.common.entities.base import Entity
from arekit.common.experiment.api.ctx_serialization import SerializationData
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.options import TextParseOptions
from arekit.processing.text.parser import DefaultTextParser


def entity_to_group_func(entity, synonyms):
    assert(isinstance(entity, Entity) or entity is None)
    assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)

    if entity is None:
        return None

    # By default, we provide the related group index.
    group_index = entity.GroupIndex
    if group_index is not None:
        return group_index

    if synonyms is None:
        return None

    # Otherwise, we search for the related group index
    # using synonyms collection.
    value = entity.Value
    if not synonyms.contains_synonym_value(value):
        return None
    return synonyms.get_synonym_group_index(value)


def create_text_parser(exp_data):

    if not isinstance(exp_data, SerializationData):
        # We do not utlize text_parser in such case.
        return None

    parse_options = TextParseOptions(parse_entities=True,
                                     stemmer=exp_data.Stemmer,
                                     frame_variants_collection=exp_data.FrameVariantCollection)

    return DefaultTextParser(parse_options=parse_options)
